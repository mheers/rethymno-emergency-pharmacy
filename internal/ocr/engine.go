// Package ocr implements local PP-OCRv6 inference (detection + recognition)
// through ONNX Runtime, with the exact preprocessing and postprocessing
// semantics of the official PaddleX ONNX exports.
//
// The engine is deliberately pure Go: it works on raw BGR buffers instead
// of gocv.Mat, because loading the ONNX Runtime shared library into a
// process that also uses gocv corrupts OpenCV allocations (verified
// empirically; see internal/ocr/worker.go for the process-splitting
// strategy).
package ocr

import (
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"log"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/yalue/onnxruntime_go"
)

// OCRLine is one recognized text line with its axis-aligned bounding box
// (in the coordinate space of the image passed to Recognize) and a mean
// character confidence in [0,1].
type OCRLine struct {
	Text       string
	Confidence float32
	Box        image.Rectangle
}

// Config configures the OCR engine. ModelPath is the directory containing
// the PP-OCRv6 ONNX models; when empty, models registered via
// SetEmbeddedModels are used instead.
type Config struct {
	ModelPath     string // directory containing PP-OCRv6_*_onnx/inference.onnx
	DetModelBytes []byte // optional explicit det model bytes (embedded mode)
	RecModelBytes []byte // optional explicit rec model bytes (embedded mode)
	DetModel      string // optional explicit path
	RecModel      string // optional explicit path
	RecModelDir   string // optional model directory name under ModelPath
	CharDict      string // optional explicit path to char_dict.json
	NoDetection   bool   // skip loading the detector for fixed-layout inputs
	DetLongSide   int    // det resize long side (960)
	DetThresh     float64
	DetBoxThresh  float64
	DetUnclip     float64
	RecHeight     int // 48
	RecMinWidth   int // 320
	BatchSize     int // rec batch (default 16)
	NumThreads    int
	Log           *log.Logger
}

// Engine owns the two ONNX sessions and the character dictionary.
type Engine struct {
	cfg         Config
	det         *onnxruntime_go.DynamicAdvancedSession
	rec         *onnxruntime_go.DynamicAdvancedSession
	chars       []string
	initialized bool
	closed      bool
}

var (
	detInputNames  = []string{"x"}
	detOutputNames = []string{"fetch_name_0"}
	recInputNames  = []string{"x"}
	recOutputNames = []string{"fetch_name_0"}
)

// nClasses is the fixed vocabulary size of the PP-OCRv6 rec model
// (18712 dictionary characters + blank + sentinel).
const nClasses = 18714

// DefaultCharDict is embedded so the engine works without a dict file.
//
//go:embed char_dict.json
var DefaultCharDict []byte

var (
	initOnce = sync.Once{}
	initErr  error
)

func defaultLog() *log.Logger { return log.New(os.Stderr, "[ocr] ", log.LstdFlags) }

// ensureRuntime initializes the ONNX Runtime environment exactly once. The
// shared library is located via PHARMA_OCR_ORT_LIB, the embedded library
// registered with SetEmbeddedLib, LD_LIBRARY_PATH or the default
// "onnxruntime.so".
func ensureRuntime() error {
	initOnce.Do(func() {
		if p := os.Getenv("PHARMA_OCR_ORT_LIB"); p != "" {
			onnxruntime_go.SetSharedLibraryPath(p)
		} else if len(embeddedLibBytes) > 0 {
			path, err := embeddedLibPath()
			if err != nil {
				initErr = fmt.Errorf("ocr: extract embedded onnxruntime: %w", err)
				return
			}
			onnxruntime_go.SetSharedLibraryPath(path)
		}
		initErr = onnxruntime_go.InitializeEnvironment()
	})
	return initErr
}

// New creates an Engine and loads the ONNX models. With ModelPath set, the
// models are resolved as <ModelPath>/<dir>/inference.onnx unless explicit
// paths are given; otherwise the bytes registered with SetEmbeddedModels
// (or DetModelBytes/RecModelBytes) are loaded. Fixed-layout callers can
// set NoDetection to avoid loading the detector.
func New(cfg Config) (*Engine, error) {
	embedded := cfg.ModelPath == ""
	detBytes, recBytes := cfg.DetModelBytes, cfg.RecModelBytes
	if embedded {
		if len(detBytes) == 0 {
			detBytes = embeddedDetBytes
		}
		if len(recBytes) == 0 {
			recBytes = embeddedRecBytes
		}
		if len(recBytes) == 0 {
			return nil, errors.New("ocr: no models: set ModelPath or run scripts/bootstrap.sh to embed them")
		}
		if !cfg.NoDetection && len(detBytes) == 0 {
			return nil, errors.New("ocr: no det model: set ModelPath or run scripts/bootstrap.sh to embed them")
		}
	}
	if cfg.Log == nil {
		cfg.Log = defaultLog()
	}
	if cfg.DetLongSide == 0 {
		cfg.DetLongSide = 960
	}
	if cfg.DetThresh == 0 {
		cfg.DetThresh = 0.3
	}
	if cfg.DetBoxThresh == 0 {
		cfg.DetBoxThresh = 0.6
	}
	if cfg.DetUnclip == 0 {
		cfg.DetUnclip = 1.5
	}
	if cfg.RecHeight == 0 {
		cfg.RecHeight = 48
	}
	if cfg.RecMinWidth == 0 {
		cfg.RecMinWidth = 320
	}
	if cfg.BatchSize == 0 {
		cfg.BatchSize = 16
	}
	if cfg.RecModelDir == "" {
		cfg.RecModelDir = "PP-OCRv6_small_rec_onnx"
	}

	if err := ensureRuntime(); err != nil {
		return nil, fmt.Errorf("ocr: initialize onnxruntime: %w", err)
	}

	e := &Engine{cfg: cfg}

	chars, err := loadCharDict(cfg.CharDict)
	if err != nil {
		return nil, err
	}
	e.chars = chars

	var detPath, recPath string
	if !cfg.NoDetection && len(detBytes) == 0 {
		detPath, err = resolveModel(cfg.ModelPath, cfg.DetModel, "PP-OCRv6_medium_det_onnx")
		if err != nil {
			return nil, err
		}
	}
	if len(recBytes) == 0 {
		recPath, err = resolveModel(cfg.ModelPath, cfg.RecModel, cfg.RecModelDir)
		if err != nil {
			return nil, err
		}
	}

	var opts *onnxruntime_go.SessionOptions
	if cfg.NumThreads > 0 {
		opts, err = onnxruntime_go.NewSessionOptions()
		if err == nil {
			_ = opts.SetIntraOpNumThreads(cfg.NumThreads)
			_ = opts.SetInterOpNumThreads(1)
			_ = opts.SetGraphOptimizationLevel(onnxruntime_go.GraphOptimizationLevelEnableAll)
			defer opts.Destroy()
		}
	}

	if !cfg.NoDetection {
		e.det, err = loadAdvancedSession(detBytes, detPath, "det", opts)
		if err != nil {
			return nil, err
		}
	}
	e.rec, err = loadAdvancedSession(recBytes, recPath, "rec", opts)
	if err != nil {
		return nil, err
	}
	e.initialized = true
	return e, nil
}

// loadAdvancedSession creates an ONNX session from bytes when available,
// falling back to the file at path. The det and rec models in this
// pipeline share the same input/output names ("x" / "fetch_name_0").
func loadAdvancedSession(data []byte, path, what string, opts *onnxruntime_go.SessionOptions) (*onnxruntime_go.DynamicAdvancedSession, error) {
	if len(data) > 0 {
		s, err := onnxruntime_go.NewDynamicAdvancedSessionWithONNXData(data, detInputNames, detOutputNames, opts)
		if err != nil {
			return nil, fmt.Errorf("ocr: load %s model: %w", what, err)
		}
		return s, nil
	}
	s, err := onnxruntime_go.NewDynamicAdvancedSession(path, detInputNames, detOutputNames, opts)
	if err != nil {
		return nil, fmt.Errorf("ocr: load %s model: %w", what, err)
	}
	return s, nil
}

func resolveModel(modelPath, explicit, dirName string) (string, error) {
	if explicit != "" {
		if _, err := os.Stat(explicit); err != nil {
			return "", fmt.Errorf("ocr: model %s: %w", explicit, err)
		}
		return explicit, nil
	}
	cands := []string{
		filepath.Join(modelPath, dirName, "inference.onnx"),
		filepath.Join(modelPath, "det.onnx"),
		filepath.Join(modelPath, "rec.onnx"),
	}
	for _, c := range cands {
		if _, err := os.Stat(c); err == nil {
			return c, nil
		}
	}
	return "", fmt.Errorf("ocr: no model found in %s (looked for %s/inference.onnx)", modelPath, dirName)
}

func loadCharDict(path string) ([]string, error) {
	var data []byte
	var err error
	if path != "" {
		data, err = os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("ocr: read char dict: %w", err)
		}
	} else {
		data = DefaultCharDict
	}
	var chars []string
	if err := json.Unmarshal(data, &chars); err != nil {
		return nil, fmt.Errorf("ocr: parse char dict: %w", err)
	}
	if len(chars) < 2 {
		return nil, fmt.Errorf("ocr: char dict too small (%d)", len(chars))
	}
	return chars, nil
}

// ProbeDet runs detection on a raw image and returns the probability map
// (diagnostic).
func ProbeDet(img *Image, modelPath string) ([]float32, error) {
	eng, err := New(Config{ModelPath: modelPath})
	if err != nil {
		return nil, err
	}
	defer eng.Close()
	h, w := img.H, img.W
	scale := 1.0
	if h > w {
		scale = float64(eng.cfg.DetLongSide) / float64(h)
	} else {
		scale = float64(eng.cfg.DetLongSide) / float64(w)
	}
	if scale >= 1.0 {
		scale = 1.0
	}
	var work *Image
	if scale < 1.0 {
		work = img.ResizeBilinear(int(float64(w)*scale), int(float64(h)*scale))
	} else {
		work = img
	}
	wh, ww := work.H, work.W
	ph := (wh + 31) / 32 * 32
	pw := (ww + 31) / 32 * 32
	padded := make([]byte, ph*pw*3)
	for y := 0; y < wh; y++ {
		copy(padded[y*pw*3:(y*pw+ww)*3], work.Data[y*ww*3:(y+1)*ww*3])
	}
	// NCHW tensor: channel planes laid out as [C][H][W]
	data := make([]float32, ph*pw*3)
	plane := ph * pw
	for y := 0; y < ph; y++ {
		for x := 0; x < pw; x++ {
			p := (y*pw + x) * 3
			off := y*pw + x
			data[off] = (float32(padded[p])/255 - 0.485) / 0.229
			data[plane+off] = (float32(padded[p+1])/255 - 0.456) / 0.224
			data[2*plane+off] = (float32(padded[p+2])/255 - 0.406) / 0.225
		}
	}
	detIn, err := onnxruntime_go.NewTensor[float32](onnxruntime_go.NewShape(1, 3, int64(ph), int64(pw)), data)
	if err != nil {
		return nil, err
	}
	defer detIn.Destroy()
	detOut, err := onnxruntime_go.NewTensor[float32](onnxruntime_go.NewShape(1, 1, int64(ph), int64(pw)), make([]float32, ph*pw))
	if err != nil {
		return nil, err
	}
	defer detOut.Destroy()
	if err := eng.det.Run([]onnxruntime_go.Value{detIn}, []onnxruntime_go.Value{detOut}); err != nil {
		return nil, err
	}
	return detOut.GetData(), nil
}

// Close releases all ONNX sessions. Must be called exactly once.
func (e *Engine) Close() error {
	if e.closed {
		return nil
	}
	e.closed = true
	if !e.initialized {
		return nil
	}
	var err1 error
	if e.det != nil {
		err1 = e.det.Destroy()
	}
	err2 := e.rec.Destroy()
	if err1 != nil {
		return err1
	}
	return err2
}

// Recognize runs detection then recognition on the given image and returns
// the recognized lines with boxes in the image coordinate space, sorted
// top-to-bottom then left-to-right.
func (e *Engine) Recognize(img *Image) ([]OCRLine, error) {
	if e.det == nil {
		return nil, errors.New("ocr: detector is disabled")
	}
	boxes, err := e.Detect(img)
	if err != nil {
		return nil, err
	}
	if len(boxes) == 0 {
		return nil, nil
	}
	lines, err := e.RecognizeBoxes(img, boxes)
	if err != nil {
		return nil, err
	}
	sort.SliceStable(lines, func(i, j int) bool {
		a, b := lines[i], lines[j]
		if a.Box.Min.Y != b.Box.Min.Y {
			return a.Box.Min.Y < b.Box.Min.Y
		}
		return a.Box.Min.X < b.Box.Min.X
	})
	return lines, nil
}

// Detect runs the PP-OCRv6 detection model with DB postprocessing and
// returns 4-point boxes in image coordinates.
func (e *Engine) Detect(img *Image) ([][4][2]float64, error) {
	if e.det == nil {
		return nil, errors.New("ocr: detector is disabled")
	}
	if img == nil || img.W == 0 || img.H == 0 {
		return nil, nil
	}
	h, w := img.H, img.W
	scale := 1.0
	if h > w {
		scale = float64(e.cfg.DetLongSide) / float64(h)
	} else {
		scale = float64(e.cfg.DetLongSide) / float64(w)
	}
	if scale >= 1.0 {
		scale = 1.0
	}

	var work *Image
	if scale < 1.0 {
		work = img.ResizeBilinear(int(float64(w)*scale), int(float64(h)*scale))
	} else {
		work = img
	}

	wh, ww := work.H, work.W
	ph := (wh + 31) / 32 * 32
	pw := (ww + 31) / 32 * 32
	// zero-padded copy
	padded := make([]byte, ph*pw*3)
	for y := 0; y < wh; y++ {
		copy(padded[y*pw*3:(y*pw+ww)*3], work.Data[y*ww*3:(y+1)*ww*3])
	}

	// NCHW float32, BGR order: (x/255 - mean)/std
	data := make([]float32, ph*pw*3)
	plane := ph * pw
	for y := 0; y < ph; y++ {
		for x := 0; x < pw; x++ {
			p := (y*pw + x) * 3
			off := y*pw + x
			data[off] = (float32(padded[p])/255 - 0.485) / 0.229
			data[plane+off] = (float32(padded[p+1])/255 - 0.456) / 0.224
			data[2*plane+off] = (float32(padded[p+2])/255 - 0.406) / 0.225
		}
	}
	detIn, err := onnxruntime_go.NewTensor[float32](onnxruntime_go.NewShape(1, 3, int64(ph), int64(pw)), data)
	if err != nil {
		return nil, fmt.Errorf("ocr: det input tensor: %w", err)
	}
	defer detIn.Destroy()

	detOut, err := onnxruntime_go.NewTensor[float32](onnxruntime_go.NewShape(1, 1, int64(ph), int64(pw)), make([]float32, ph*pw))
	if err != nil {
		return nil, fmt.Errorf("ocr: det output tensor: %w", err)
	}
	defer detOut.Destroy()

	if err := e.det.Run([]onnxruntime_go.Value{detIn}, []onnxruntime_go.Value{detOut}); err != nil {
		return nil, fmt.Errorf("ocr: det inference: %w", err)
	}

	return dbPostprocess(detOut.GetData(), ph, pw, e.cfg, scale), nil
}

// RecognizeBoxes runs the recognition model on crops of img given by boxes
// (4-point rectangles) and returns OCRLine results in the image coordinate
// space.
func (e *Engine) RecognizeBoxes(img *Image, boxes [][4][2]float64) ([]OCRLine, error) {
	crops := make([]cropItem, 0, len(boxes))
	widths := make([]int, 0, len(boxes))
	for _, b := range boxes {
		crop, ok := e.cropFromBox(img, b)
		if !ok {
			continue
		}
		if crop.W < 16 {
			continue
		}
		crops = append(crops, cropItem{line: OCRLine{Box: bboxOf(b)}, img: crop})
		widths = append(widths, crop.W)
	}

	results := make([]OCRLine, len(crops))
	for start := 0; start < len(crops); start += e.cfg.BatchSize {
		end := start + e.cfg.BatchSize
		if end > len(crops) {
			end = len(crops)
		}
		if err := e.recognizeBatch(crops[start:end], widths[start:end], results[start:end]); err != nil {
			return nil, err
		}
	}
	return results, nil
}

type cropItem struct {
	line OCRLine
	img  *Image
}

func (e *Engine) recognizeBatch(batch []cropItem, widths []int, results []OCRLine) error {
	n := len(batch)
	maxW := e.cfg.RecMinWidth
	for i, c := range batch {
		aw := e.resizeWidth(c.img.W, c.img.H)
		widths[i] = aw
		if aw > maxW {
			maxW = aw
		}
	}
	maxW = (maxW + 7) / 8 * 8
	seqLen := maxW / 8

	data := make([]float32, n*3*e.cfg.RecHeight*maxW)
	for i, c := range batch {
		aw := widths[i]
		resized := c.img.ResizeBilinear(aw, e.cfg.RecHeight)
		base := i * 3 * e.cfg.RecHeight * maxW
		plane := e.cfg.RecHeight * maxW
		for y := 0; y < e.cfg.RecHeight; y++ {
			for x := 0; x < aw; x++ {
				p := (y*aw + x) * 3
				off := base + y*maxW + x
				data[off] = (float32(resized.Data[p])/255 - 0.5) / 0.5
				data[off+plane] = (float32(resized.Data[p+1])/255 - 0.5) / 0.5
				data[off+2*plane] = (float32(resized.Data[p+2])/255 - 0.5) / 0.5
			}
		}
	}
	recIn, err := onnxruntime_go.NewTensor[float32](onnxruntime_go.NewShape(int64(n), 3, int64(e.cfg.RecHeight), int64(maxW)), data)
	if err != nil {
		return fmt.Errorf("ocr: rec input tensor: %w", err)
	}
	defer recIn.Destroy()

	recOut, err := onnxruntime_go.NewTensor[float32](onnxruntime_go.NewShape(int64(n), int64(seqLen), nClasses), make([]float32, n*seqLen*nClasses))
	if err != nil {
		return fmt.Errorf("ocr: rec output tensor: %w", err)
	}
	defer recOut.Destroy()

	if err := e.rec.Run([]onnxruntime_go.Value{recIn}, []onnxruntime_go.Value{recOut}); err != nil {
		return fmt.Errorf("ocr: rec inference: %w", err)
	}

	out := recOut.GetData()
	step := seqLen * nClasses
	for i, c := range batch {
		text, conf := e.ctcDecode(out[i*step : (i+1)*step])
		results[i] = OCRLine{Text: text, Confidence: conf, Box: c.line.Box}
	}
	return nil
}

// resizeWidth computes the recognition-model width for a crop: fixed height
// 48, aspect-preserving width, minimum RecMinWidth.
func (e *Engine) resizeWidth(w, h int) int {
	aw := int(float64(e.cfg.RecHeight)*float64(w)/float64(h)) + 1
	if aw < e.cfg.RecMinWidth {
		aw = e.cfg.RecMinWidth
	}
	return aw
}

// cropFromBox extracts the axis-aligned bounding box of a 4-point box.
func (e *Engine) cropFromBox(img *Image, box [4][2]float64) (*Image, bool) {
	r := bboxOf(box)
	ix0, iy0, ix1, iy1 := r.Min.X, r.Min.Y, r.Max.X, r.Max.Y
	if ix0 < 0 {
		ix0 = 0
	}
	if iy0 < 0 {
		iy0 = 0
	}
	if ix1 >= img.W {
		ix1 = img.W - 1
	}
	if iy1 >= img.H {
		iy1 = img.H - 1
	}
	if ix1 <= ix0 || iy1 <= iy0 {
		return nil, false
	}
	crop := img.Crop(ix0, iy0, ix1+1, iy1+1)
	if crop.W == 0 {
		return nil, false
	}
	return crop, true
}

func bboxOf(box [4][2]float64) image.Rectangle {
	x0, y0 := box[0][0], box[0][1]
	x1, y1 := box[0][0], box[0][1]
	for _, p := range box[1:] {
		if p[0] < x0 {
			x0 = p[0]
		}
		if p[0] > x1 {
			x1 = p[0]
		}
		if p[1] < y0 {
			y0 = p[1]
		}
		if p[1] > y1 {
			y1 = p[1]
		}
	}
	return image.Rect(int(x0), int(y0), int(x1)+1, int(y1)+1)
}

// ctcDecode collapses the CTC output. Class 0 is the blank symbol; the
// official ONNX exports offset the character dictionary by one, so output
// class c (c >= 1) maps to dict[c-1].
func (e *Engine) ctcDecode(out []float32) (string, float32) {
	var b []byte
	var total float32
	n := 0
	prev := -1
	for i := 0; i+nClasses <= len(out); i += nClasses {
		row := out[i : i+nClasses]
		best := float32(-1)
		bestIdx := 0
		for j := 0; j < nClasses; j++ {
			if row[j] > best {
				best = row[j]
				bestIdx = j
			}
		}
		if bestIdx == 0 {
			prev = -1
			continue
		}
		ci := bestIdx - 1
		if ci < 0 || ci >= len(e.chars) {
			prev = -1
			continue
		}
		if ci != prev {
			b = append(b, e.chars[ci]...)
			total += best
			n++
		}
		prev = ci
	}
	if n == 0 {
		return "", 0
	}
	return string(b), total / float32(n)
}
