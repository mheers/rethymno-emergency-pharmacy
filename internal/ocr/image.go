package ocr

// Image is a pure-Go BGR image buffer. The OCR engine deliberately avoids
// gocv: loading the ONNX Runtime library into a process that also uses
// gocv corrupts OpenCV allocations on this platform (verified empirically),
// so the OCR stage runs either in the same process on raw buffers or in a
// dedicated worker process.

// Image is a BGR image with explicit dimensions.
type Image struct {
	Data []byte // BGR bytes, row-major
	W, H int
}

// NewImage creates an Image from raw bytes.
func NewImage(data []byte, w, h int) *Image {
	return &Image{Data: data, W: w, H: h}
}

// GrayToBGR expands a single-channel image to BGR.
func GrayToBGR(gray []byte, w, h int) *Image {
	out := make([]byte, len(gray)*3)
	for i, v := range gray {
		out[i*3] = v
		out[i*3+1] = v
		out[i*3+2] = v
	}
	return &Image{Data: out, W: w, H: h}
}

// Crop returns the axis-aligned region [x0,x1)×[y0,y1) as a new Image.
func (im *Image) Crop(x0, y0, x1, y1 int) *Image {
	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}
	if x1 > im.W {
		x1 = im.W
	}
	if y1 > im.H {
		y1 = im.H
	}
	if x1 <= x0 || y1 <= y0 {
		return &Image{Data: nil, W: 0, H: 0}
	}
	w := x1 - x0
	h := y1 - y0
	out := make([]byte, w*h*3)
	for y := 0; y < h; y++ {
		src := (y+y0)*im.W*3 + x0*3
		dst := y * w * 3
		copy(out[dst:dst+w*3], im.Data[src:src+w*3])
	}
	return &Image{Data: out, W: w, H: h}
}

// ResizeBilinear resizes the image to dstW×dstH using bilinear
// interpolation (matches OpenCV INTER_LINEAR closely enough for OCR).
func (im *Image) ResizeBilinear(dstW, dstH int) *Image {
	out := make([]byte, dstW*dstH*3)
	if im.W == 0 || im.H == 0 || dstW == 0 || dstH == 0 {
		return &Image{Data: out, W: dstW, H: dstH}
	}
	xr := float64(im.W) / float64(dstW)
	yr := float64(im.H) / float64(dstH)
	for dy := 0; dy < dstH; dy++ {
		sy := float64(dy) * yr
		y0 := int(sy)
		if y0 > im.H-2 {
			y0 = im.H - 2
		}
		fy := sy - float64(y0)
		for dx := 0; dx < dstW; dx++ {
			sx := float64(dx) * xr
			x0 := int(sx)
			if x0 > im.W-2 {
				x0 = im.W - 2
			}
			fx := sx - float64(x0)
			for c := 0; c < 3; c++ {
				p00 := float64(im.Data[(y0*im.W+x0)*3+c])
				p10 := float64(im.Data[(y0*im.W+x0+1)*3+c])
				p01 := float64(im.Data[((y0+1)*im.W+x0)*3+c])
				p11 := float64(im.Data[((y0+1)*im.W+x0+1)*3+c])
				top := p00 + (p10-p00)*fx
				bot := p01 + (p11-p01)*fx
				out[(dy*dstW+dx)*3+c] = byte(top + (bot-top)*fy + 0.5)
			}
		}
	}
	return &Image{Data: out, W: dstW, H: dstH}
}

// ResizeArea downsizes using area averaging (equivalent to INTER_AREA for
// downscaling; falls back to nearest for upscaling).
func (im *Image) ResizeArea(dstW, dstH int) *Image {
	if dstW >= im.W || dstH >= im.H {
		return im.ResizeBilinear(dstW, dstH)
	}
	out := make([]byte, dstW*dstH*3)
	xr := float64(im.W) / float64(dstW)
	yr := float64(im.H) / float64(dstH)
	for dy := 0; dy < dstH; dy++ {
		y0 := int(float64(dy) * yr)
		y1 := int(float64(dy+1) * yr)
		if y1 <= y0 {
			y1 = y0 + 1
		}
		for dx := 0; dx < dstW; dx++ {
			x0 := int(float64(dx) * xr)
			x1 := int(float64(dx+1) * xr)
			if x1 <= x0 {
				x1 = x0 + 1
			}
			for c := 0; c < 3; c++ {
				var sum int
				var n int
				for y := y0; y < y1; y++ {
					row := y * im.W * 3
					for x := x0; x < x1; x++ {
						sum += int(im.Data[(row+x*3)+c])
						n++
					}
				}
				if n == 0 {
					continue
				}
				out[(dy*dstW+dx)*3+c] = byte(sum / n)
			}
		}
	}
	return &Image{Data: out, W: dstW, H: dstH}
}
