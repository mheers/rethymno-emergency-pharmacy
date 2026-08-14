package ocr

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"io"
	"log"
	"os"
	"os/exec"
	"sync"
)

// The OCR engine must not share a process with gocv: loading the ONNX
// Runtime shared library corrupts OpenCV allocations (verified
// empirically on glibc 2.36+ with Go 1.24/1.25). The pipeline therefore
// runs the OCR stage in a dedicated worker subprocess: the parent sends
// PNG-encoded column images over stdin, the worker answers with JSON OCR
// lines over stdout.
//
// Protocol (length-prefixed frames, 4-byte big-endian):
//
//	request:  [u32 len][png bytes]
//	response: [u32 len][json array]
//
// JSON item: {"text": "...", "conf": 0.99, "x0": 1, "y0": 2, "x1": 3, "y1": 4}
// (boxes in the coordinate space of the sent image)

// wireLine is the JSON shape exchanged with the worker.
type wireLine struct {
	Text string  `json:"text"`
	Conf float32 `json:"conf"`
	X0   int     `json:"x0"`
	Y0   int     `json:"y0"`
	X1   int     `json:"x1"`
	Y1   int     `json:"y1"`
}

// Remote is a client for the OCR worker process. It satisfies the same
// Recognize interface as Engine but runs inference in a child process.
type Remote struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	stderr io.ReadCloser
	once   sync.Once
	mu     sync.Mutex
}

// StartRemote spawns the OCR worker (by default this executable's
// "ocr-worker" subcommand, or any command that wires up WorkerMain) with
// the given model directory. An empty modelsDir selects the embedded
// models; recModel selects the embedded recognition model variant.
func StartRemote(cmd []string, modelsDir, recModel string, numThreads int) (*Remote, error) {
	if len(cmd) == 0 {
		return nil, fmt.Errorf("ocr: worker command is empty")
	}
	args := append([]string{}, cmd[1:]...)
	if modelsDir != "" {
		args = append(args, "--models", modelsDir)
	}
	if recModel != "" && recModel != "small" {
		args = append(args, "--rec-model", recModel)
	}
	if numThreads > 0 {
		args = append(args, "--threads", fmt.Sprint(numThreads))
	}
	proc := exec.Command(cmd[0], args...)
	stdin, err := proc.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("ocr: worker stdin: %w", err)
	}
	stdout, err := proc.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("ocr: worker stdout: %w", err)
	}
	stderr, err := proc.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("ocr: worker stderr: %w", err)
	}
	if err := proc.Start(); err != nil {
		return nil, fmt.Errorf("ocr: start worker: %w", err)
	}
	r := &Remote{
		cmd:    proc,
		stdin:  stdin,
		stdout: bufio.NewReader(stdout),
		stderr: stderr,
	}
	go func() {
		sc := bufio.NewScanner(r.stderr)
		for sc.Scan() {
			log.Printf("[ocr-worker] %s", sc.Text())
		}
	}()
	return r, nil
}

// Recognize sends the image (as PNG) to the worker and returns the OCR
// lines in the image coordinate space.
func (r *Remote) Recognize(img *Image) ([]OCRLine, error) {
	return r.recognize(img)
}

// RecognizeColumn uses fixed-layout row segmentation in the worker.
func (r *Remote) RecognizeColumn(img *Image) ([]OCRLine, error) {
	return r.recognize(img)
}

func (r *Remote) recognize(img *Image) ([]OCRLine, error) {
	if img == nil || img.W == 0 || img.H == 0 {
		return nil, nil
	}
	pngBytes, err := encodePNG(img)
	if err != nil {
		return nil, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(pngBytes)))
	if _, err := r.stdin.Write(hdr[:]); err != nil {
		return nil, fmt.Errorf("ocr: worker write: %w", err)
	}
	if _, err := r.stdin.Write(pngBytes); err != nil {
		return nil, fmt.Errorf("ocr: worker write: %w", err)
	}

	var respHdr [4]byte
	if _, err := io.ReadFull(r.stdout, respHdr[:]); err != nil {
		return nil, fmt.Errorf("ocr: worker read: %w", err)
	}
	n := binary.BigEndian.Uint32(respHdr[:])
	if n == 0 || n > 64<<20 {
		return nil, fmt.Errorf("ocr: worker bad frame length %d", n)
	}
	payload := make([]byte, n)
	if _, err := io.ReadFull(r.stdout, payload); err != nil {
		return nil, fmt.Errorf("ocr: worker read payload: %w", err)
	}
	var items []wireLine
	if err := json.Unmarshal(payload, &items); err != nil {
		return nil, fmt.Errorf("ocr: worker json: %w", err)
	}
	out := make([]OCRLine, 0, len(items))
	for _, it := range items {
		out = append(out, OCRLine{
			Text:       it.Text,
			Confidence: it.Conf,
			Box:        image.Rect(it.X0, it.Y0, it.X1, it.Y1),
		})
	}
	return out, nil
}

// Close terminates the worker.
func (r *Remote) Close() error {
	var err error
	r.once.Do(func() {
		_ = r.stdin.Close()
		err = r.cmd.Wait()
	})
	return err
}

func encodePNG(img *Image) ([]byte, error) {
	var buf bytes.Buffer
	// convert to gray for compact encoding
	gray := make([]byte, img.W*img.H)
	if len(img.Data) == img.W*img.H {
		copy(gray, img.Data)
	} else {
		for i := 0; i < img.W*img.H; i++ {
			gray[i] = img.Data[i*3]
		}
	}
	gi := &image.Gray{Pix: gray, Stride: img.W, Rect: image.Rect(0, 0, img.W, img.H)}
	if err := png.Encode(&buf, gi); err != nil {
		return nil, fmt.Errorf("ocr: png encode: %w", err)
	}
	return buf.Bytes(), nil
}

// RunWorker serves fixed-layout OCR requests until stdin closes.
func RunWorker(modelsDir string, numThreads int) error {
	if numThreads <= 0 {
		numThreads = 4
	}
	eng, err := New(Config{
		ModelPath:   modelsDir,
		NumThreads:  numThreads,
		BatchSize:   16,
		NoDetection: true,
	})
	if err != nil {
		return err
	}
	defer eng.Close()
	log.Printf("ocr-worker ready (models=%s)", modelsDir)

	in := bufio.NewReader(os.Stdin)
	out := bufio.NewWriter(os.Stdout)
	for {
		var hdr [4]byte
		if _, err := io.ReadFull(in, hdr[:]); err != nil {
			if err == io.EOF {
				return nil
			}
			return fmt.Errorf("ocr-worker: read header: %w", err)
		}
		n := binary.BigEndian.Uint32(hdr[:])
		if n == 0 || n > 64<<20 {
			return fmt.Errorf("ocr-worker: bad frame length %d", n)
		}
		payload := make([]byte, n)
		if _, err := io.ReadFull(in, payload); err != nil {
			return fmt.Errorf("ocr-worker: read payload: %w", err)
		}
		img, err := decodePNG(payload)
		if err != nil {
			return fmt.Errorf("ocr-worker: decode: %w", err)
		}
		lines, err := eng.RecognizeColumn(img)
		if err != nil {
			log.Printf("ocr-worker: recognize: %v", err)
			lines = nil
		}
		resp, err := json.Marshal(linesToWire(lines))
		if err != nil {
			return fmt.Errorf("ocr-worker: marshal: %w", err)
		}
		binary.BigEndian.PutUint32(hdr[:], uint32(len(resp)))
		if _, err := out.Write(hdr[:]); err != nil {
			return fmt.Errorf("ocr-worker: write header: %w", err)
		}
		if _, err := out.Write(resp); err != nil {
			return fmt.Errorf("ocr-worker: write payload: %w", err)
		}
		if err := out.Flush(); err != nil {
			return fmt.Errorf("ocr-worker: flush: %w", err)
		}
	}
}

func decodePNG(payload []byte) (*Image, error) {
	im, err := png.Decode(bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	bounds := im.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	bgr := make([]byte, w*h*3)
	i := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := im.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			bgr[i] = byte(b >> 8)
			bgr[i+1] = byte(g >> 8)
			bgr[i+2] = byte(r >> 8)
			i += 3
		}
	}
	return &Image{Data: bgr, W: w, H: h}, nil
}

func linesToWire(lines []OCRLine) []wireLine {
	out := make([]wireLine, 0, len(lines))
	for _, l := range lines {
		out = append(out, wireLine{
			Text: l.Text,
			Conf: l.Confidence,
			X0:   l.Box.Min.X,
			Y0:   l.Box.Min.Y,
			X1:   l.Box.Max.X,
			Y1:   l.Box.Max.Y,
		})
	}
	return out
}
