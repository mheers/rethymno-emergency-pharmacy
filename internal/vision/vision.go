// Package vision implements the OpenCV preprocessing stage of the
// pharmacy-schedule OCR pipeline: decode, geometry inspection, perspective
// correction, deskew, grid detection, column segmentation, grid removal
// and OCR-variant generation. All work happens on gocv.Mat with explicit
// ownership and defer Close().
package vision

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"sort"

	"gocv.io/x/gocv"
)

var white = color.RGBA{R: 255, G: 255, B: 255, A: 255}

// Config tunes the preprocessing pipeline.
type Config struct {
	TargetDPI         int     // nominal DPI used for resolution estimates (default 150)
	MaxWidth          int     // downscale images wider than this before processing (default 4096)
	AdaptiveBlockSize int     // adaptive threshold block size (default 31)
	AdaptiveC         float64 // adaptive threshold constant (default 10)
	Deskew            bool    // enable projection-profile deskew (default true)
	RemoveGridLines   bool    // produce text-only variants (default true)
	DetectColumns     bool    // segment weekday columns (default true)
	GridLineMinLen    float64 // min line length as fraction of table size (default 0.5)
}

// Column is one segmented table column.
type Column struct {
	Index int
	X0    int
	X1    int
	Width int
	Rect  image.Rectangle
	Image gocv.Mat // text-optimized column crop (grid removed when enabled)
}

// Geometry describes the measured document geometry.
type Geometry struct {
	Width                int
	Height               int
	Channels             int
	AspectRatio          float64
	EffectiveDPI         int
	PerspectiveCorrected bool
	RotationDegrees      float64
	ContentRect          image.Rectangle // table area after correction
}

// ProcessedSchedule is the result of the preprocessing stage.
type ProcessedSchedule struct {
	Original    gocv.Mat
	Corrected   gocv.Mat // perspective-corrected and deskewed (== Original when no correction)
	Geometry    Geometry
	Columns     []Column
	DebugImages map[string]gocv.Mat
}

// Processor runs the vision pipeline.
type Processor struct {
	Config Config
}

// New creates a Processor with defaults applied.
func New(cfg Config) *Processor {
	if cfg.TargetDPI == 0 {
		cfg.TargetDPI = 150
	}
	if cfg.MaxWidth == 0 {
		cfg.MaxWidth = 4096
	}
	if cfg.AdaptiveBlockSize == 0 {
		cfg.AdaptiveBlockSize = 31
	}
	if cfg.AdaptiveC == 0 {
		cfg.AdaptiveC = 10
	}
	if cfg.GridLineMinLen == 0 {
		cfg.GridLineMinLen = 0.5
	}
	if !cfg.Deskew {
		cfg.Deskew = true
	}
	if !cfg.RemoveGridLines {
		cfg.RemoveGridLines = true
	}
	if !cfg.DetectColumns {
		cfg.DetectColumns = true
	}
	return &Processor{Config: cfg}
}

// Decode loads an image from raw bytes using OpenCV's decoder.
func (p *Processor) Decode(data []byte) (gocv.Mat, error) {
	img, err := gocv.IMDecode(data, gocv.IMReadColor)
	if err != nil {
		return gocv.Mat{}, fmt.Errorf("vision: decode image: %w", err)
	}
	if img.Empty() {
		img.Close()
		return gocv.Mat{}, fmt.Errorf("vision: decoded image is empty")
	}
	return img, nil
}

// Process runs the full preprocessing pipeline on the given image and
// returns the corrected image, detected columns and debug images. The
// caller owns the returned Mats (ProcessedSchedule.Original is a clone).
func (p *Processor) Process(img gocv.Mat) (*ProcessedSchedule, error) {
	s := &ProcessedSchedule{
		Original:    img.Clone(),
		DebugImages: map[string]gocv.Mat{},
	}
	work := img.Clone()
	defer work.Close()

	// Normalize excessively large images.
	if work.Cols() > p.Config.MaxWidth {
		scale := float64(p.Config.MaxWidth) / float64(work.Cols())
		dst := gocv.NewMat()
		gocv.Resize(work, &dst, image.Point{X: p.Config.MaxWidth, Y: int(float64(work.Rows()) * scale)}, 0, 0, gocv.InterpolationArea)
		work.Close()
		work = dst
	}

	g := Geometry{
		Width:        work.Cols(),
		Height:       work.Rows(),
		Channels:     work.Channels(),
		AspectRatio:  float64(work.Cols()) / float64(work.Rows()),
		EffectiveDPI: p.Config.TargetDPI,
	}

	// 1. Perspective correction (bypass for already-rectangular documents).
	perspApplied := false
	isRectangular := p.IsRectangular(work)
	if isRectangular {
		g.ContentRect = p.documentRect(work)
	} else if rect, ok := p.detectQuadrilateral(work); ok {
		corrected := p.warp(work, rect)
		work.Close()
		work = corrected
		perspApplied = true
		g.ContentRect = image.Rect(0, 0, work.Cols(), work.Rows())
		s.DebugImages["perspective.jpg"] = work.Clone()
	}
	g.PerspectiveCorrected = perspApplied

	// 2. Deskew via projection-profile optimization.
	// A rectangular rasterized schedule is already axis-aligned. Avoid the
	// expensive 25-angle projection scan for this common input class.
	if p.Config.Deskew && !isRectangular {
		angle := p.deskewAngle(work)
		g.RotationDegrees = angle
		if math.Abs(angle) > 0.5 {
			dst := gocv.NewMat()
			center := image.Pt(work.Cols()/2, work.Rows()/2)
			m := gocv.GetRotationMatrix2D(center, angle, 1.0)
			gocv.WarpAffineWithParams(work, &dst, m, image.Point{X: work.Cols(), Y: work.Rows()}, gocv.InterpolationLinear, gocv.BorderConstant, white)
			m.Close()
			work.Close()
			work = dst
			s.DebugImages["deskewed.jpg"] = work.Clone()
		}
	}

	s.Corrected = work.Clone()

	// 3. Grid detection + column segmentation.
	if p.Config.DetectColumns {
		grid := p.detectGrid(work)
		cols := p.segmentColumns(work, grid)
		for i := range cols {
			col := &cols[i]
			col.Image = p.buildTextVariant(work, col.Rect, i, s.DebugImages)
			s.Columns = append(s.Columns, *col)
		}
	}

	if s.DebugImages != nil {
		s.DebugImages["grayscale.jpg"] = p.grayscale(work)
		th := gocv.NewMat()
		gocv.AdaptiveThreshold(s.DebugImages["grayscale.jpg"], &th, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinary, p.Config.AdaptiveBlockSize, float32(p.Config.AdaptiveC))
		s.DebugImages["threshold.jpg"] = th
	}
	return s, nil
}

// SavePNG writes a Mat to a PNG file (debug helper).
func SavePNG(m gocv.Mat, path string) bool {
	return gocv.IMWrite(path, m)
}

// grayscale converts to 8-bit gray.
func (p *Processor) grayscale(src gocv.Mat) gocv.Mat {
	gray := gocv.NewMat()
	switch src.Channels() {
	case 1:
		return src.Clone()
	case 3:
		gocv.CvtColor(src, &gray, gocv.ColorBGRToGray)
	case 4:
		gocv.CvtColor(src, &gray, gocv.ColorBGRAToGray)
	default:
		gray = src.Clone()
	}
	return gray
}

// binaryInverted returns an Otsu-thresholded inverted binary image.
func (p *Processor) binaryInverted(src gocv.Mat) gocv.Mat {
	gray := p.grayscale(src)
	defer gray.Close()
	bw := gocv.NewMat()
	gocv.Threshold(gray, &bw, 0, 255, gocv.ThresholdBinaryInv+gocv.ThresholdOtsu)
	return bw
}

// IsRectangular decides whether the document is already an axis-aligned
// scan/screenshot: the four border margins must be essentially blank.
func (p *Processor) IsRectangular(img gocv.Mat) bool {
	gray := p.grayscale(img)
	defer gray.Close()
	h, w := gray.Rows(), gray.Cols()
	margin := 12
	if w < 2*margin || h < 2*margin {
		return true
	}
	// margins must be > 90% white
	check := func(roi image.Rectangle) bool {
		sub := gray.Region(roi)
		defer sub.Close()
		mean := gocv.NewMat()
		std := gocv.NewMat()
		gocv.MeanStdDev(sub, &mean, &std)
		v := mean.GetDoubleAt(0, 0)
		mean.Close()
		std.Close()
		return v > 230
	}
	if !check(image.Rect(0, 0, w, margin)) || !check(image.Rect(0, 0, margin, h)) {
		return false
	}
	if !check(image.Rect(0, h-margin, w, h)) || !check(image.Rect(w-margin, 0, w, h)) {
		return false
	}
	return true
}

// documentRect returns the content bounding box (non-blank pixels).
func (p *Processor) documentRect(img gocv.Mat) image.Rectangle {
	bw := p.binaryInverted(img)
	defer bw.Close()
	pixels := bw.ToBytes()
	minX, minY := img.Cols(), img.Rows()
	maxX, maxY := 0, 0
	found := false
	for y := 0; y < bw.Rows(); y++ {
		for x := 0; x < bw.Cols(); x++ {
			value := byte(0)
			if len(pixels) >= bw.Rows()*bw.Cols() {
				value = pixels[y*bw.Cols()+x]
			} else {
				value = bw.GetUCharAt(y, x)
			}
			if value > 0 {
				if !found {
					minX, minY = x, y
					found = true
				}
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}
	if !found {
		return image.Rect(0, 0, img.Cols(), img.Rows())
	}
	return image.Rect(minX, minY, maxX+1, maxY+1)
}

// detectQuadrilateral finds the largest plausible document quadrilateral
// via Canny + contours + ApproxPolyDP and returns its ordered corners.
func (p *Processor) detectQuadrilateral(img gocv.Mat) (image.Rectangle, bool) {
	gray := p.grayscale(img)
	defer gray.Close()
	blur := gocv.NewMat()
	gocv.GaussianBlur(gray, &blur, image.Pt(5, 5), 0, 0, gocv.BorderDefault)
	defer blur.Close()
	edges := gocv.NewMat()
	gocv.Canny(blur, &edges, 75, 200)
	defer edges.Close()

	contours := gocv.FindContours(edges, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours.Close()

	imgArea := float64(img.Rows() * img.Cols())
	bestIdx := -1
	bestArea := 0.0
	var bestPts []image.Point
	for i := 0; i < contours.Size(); i++ {
		c := contours.At(i)
		area := math.Abs(gocv.ContourArea(c))
		if area < 0.1*imgArea {
			continue
		}
		peri := gocv.ArcLength(c, true)
		approx := gocv.ApproxPolyDP(c, 0.02*peri, true)
		if approx.Size() == 4 {
			if a := math.Abs(gocv.ContourArea(approx)); a > bestArea {
				bestArea = a
				bestIdx = i
				bestPts = approx.ToPoints()
			}
		}
		approx.Close()
	}
	if bestIdx < 0 || len(bestPts) != 4 {
		return image.Rectangle{}, false
	}
	pts := bestPts

	// order: top-left, top-right, bottom-right, bottom-left
	rect := image.Rectangle{
		Min: image.Point{X: pts[0].X, Y: pts[0].Y},
		Max: image.Point{X: pts[0].X, Y: pts[0].Y},
	}
	for _, p := range pts[1:] {
		if p.X < rect.Min.X {
			rect.Min.X = p.X
		}
		if p.Y < rect.Min.Y {
			rect.Min.Y = p.Y
		}
		if p.X > rect.Max.X {
			rect.Max.X = p.X
		}
		if p.Y > rect.Max.Y {
			rect.Max.Y = p.Y
		}
	}
	return rect, true
}

// warp performs a perspective transform of the detected quadrilateral to a
// rectangle. For near-rectangular quadrilaterals this degenerates to a
// rotation-free crop via GetPerspectiveTransform.
func (p *Processor) warp(img gocv.Mat, rect image.Rectangle) gocv.Mat {
	src := gocv.NewPoint2fVectorFromPoints([]gocv.Point2f{
		{X: float32(rect.Min.X), Y: float32(rect.Min.Y)},
		{X: float32(rect.Max.X), Y: float32(rect.Min.Y)},
		{X: float32(rect.Max.X), Y: float32(rect.Max.Y)},
		{X: float32(rect.Min.X), Y: float32(rect.Max.Y)},
	})
	dst := gocv.NewPoint2fVectorFromPoints([]gocv.Point2f{
		{X: 0, Y: 0},
		{X: float32(rect.Dx()), Y: 0},
		{X: float32(rect.Dx()), Y: float32(rect.Dy())},
		{X: 0, Y: float32(rect.Dy())},
	})
	M := gocv.GetPerspectiveTransform2f(src, dst)
	defer M.Close()
	out := gocv.NewMat()
	gocv.WarpPerspectiveWithParams(img, &out, M, image.Point{X: rect.Dx(), Y: rect.Dy()}, gocv.InterpolationLinear, gocv.BorderConstant, white)
	return out
}

// deskewAngle estimates residual rotation by maximizing the variance of the
// horizontal projection profile over a small rotation range.
func (p *Processor) deskewAngle(img gocv.Mat) float64 {
	bw := p.binaryInverted(img)
	defer bw.Close()
	// crop to content to ignore blank margins
	h, w := bw.Rows(), bw.Cols()
	bestAngle := 0.0
	bestScore := -1.0
	step := 0.5
	for a := -6.0; a <= 6.0+1e-9; a += step {
		var rot = gocv.NewMat()
		if math.Abs(a) < 1e-9 {
			bw.CopyTo(&rot)
		} else {
			center := image.Pt(w/2, h/2)
			m := gocv.GetRotationMatrix2D(center, a, 1.0)
			gocv.WarpAffineWithParams(bw, &rot, m, image.Point{X: w, Y: h}, gocv.InterpolationNearestNeighbor, gocv.BorderConstant, color.RGBA{})
			m.Close()
		}
		score := projectionScore(rot)
		rot.Close()
		if score > bestScore {
			bestScore = score
			bestAngle = a
		}
	}
	return bestAngle
}

// projectionScore returns the variance of the horizontal projection of dark
// pixels; text lines produce pronounced peaks, so the variance is maximal
// when the text is aligned with the axes.
func projectionScore(bw gocv.Mat) float64 {
	rows := bw.Rows()
	cols := bw.Cols()
	pixels := bw.ToBytes()
	profile := make([]float64, rows)
	total := 0.0
	for y := 0; y < rows; y++ {
		var cnt float64
		for x := 0; x < cols; x++ {
			value := byte(0)
			if len(pixels) >= rows*cols {
				value = pixels[y*cols+x]
			} else {
				value = bw.GetUCharAt(y, x)
			}
			if value > 0 {
				cnt++
			}
		}
		profile[y] = cnt
		total += cnt
	}
	if total == 0 {
		return 0
	}
	mean := total / float64(rows)
	var variance float64
	for _, v := range profile {
		d := v - mean
		variance += d * d
	}
	return variance / float64(rows)
}

// GridLines holds detected line positions (in pixels) and the table extent.
type GridLines struct {
	VerticalX   []int
	HorizontalY []int
	Rect        image.Rectangle
}

// detectGrid finds horizontal and vertical grid lines via morphology.
func (p *Processor) detectGrid(img gocv.Mat) GridLines {
	bw := p.binaryInverted(img)
	defer bw.Close()
	h, w := bw.Rows(), bw.Cols()

	minLen := int(float64(w) * p.Config.GridLineMinLen)
	horizKernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(minLen, 1))
	vertKernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(1, minLen))
	defer horizKernel.Close()
	defer vertKernel.Close()

	horiz := gocv.NewMat()
	vert := gocv.NewMat()
	gocv.MorphologyEx(bw, &horiz, gocv.MorphOpen, horizKernel)
	gocv.MorphologyEx(bw, &vert, gocv.MorphOpen, vertKernel)
	defer horiz.Close()
	defer vert.Close()

	grid := GridLines{
		VerticalX:   linePositions(vert, w, h, true),
		HorizontalY: linePositions(horiz, w, h, false),
	}
	grid.Rect = p.documentRect(img)

	// refine: keep lines inside the content rect, drop duplicates
	grid.VerticalX = filterLines(grid.VerticalX, grid.Rect.Min.X, grid.Rect.Max.X)
	grid.HorizontalY = filterLines(grid.HorizontalY, grid.Rect.Min.Y, grid.Rect.Max.Y)
	return grid
}

// linePositions finds 1px-thick line positions by counting dark pixels per
// row/column and clustering contiguous runs that span most of the axis.
func linePositions(bw gocv.Mat, w, h int, vertical bool) []int {
	pixels := bw.ToBytes()
	length := h
	if vertical {
		length = w
	}
	threshold := float64(length) * 0.55
	pos := make([]int, 0, 16)
	for i := 0; i < length; i++ {
		var cnt int
		if vertical {
			for y := 0; y < h; y++ {
				value := byte(0)
				if len(pixels) >= w*h {
					value = pixels[y*w+i]
				} else {
					value = bw.GetUCharAt(y, i)
				}
				if value > 0 {
					cnt++
				}
			}
		} else {
			for x := 0; x < w; x++ {
				value := byte(0)
				if len(pixels) >= w*h {
					value = pixels[i*w+x]
				} else {
					value = bw.GetUCharAt(i, x)
				}
				if value > 0 {
					cnt++
				}
			}
		}
		if float64(cnt) >= threshold {
			pos = append(pos, i)
		}
	}
	return clusterPositions(pos)
}

// clusterPositions merges adjacent 1px runs into single line centers.
func clusterPositions(runs []int) []int {
	if len(runs) == 0 {
		return nil
	}
	var out []int
	start := runs[0]
	prev := runs[0]
	for _, r := range runs[1:] {
		if r-prev > 2 {
			out = append(out, (start+prev)/2)
			start = r
		}
		prev = r
	}
	out = append(out, (start+prev)/2)
	return out
}

// filterLines drops lines outside the content rect and merges close pairs.
func filterLines(lines []int, min, max int) []int {
	if len(lines) == 0 {
		return nil
	}
	sort.Ints(lines)
	var out []int
	for _, l := range lines {
		if l < min+5 || l > max-5 {
			continue
		}
		if len(out) > 0 && l-out[len(out)-1] < 30 {
			// merge nearly-coincident lines (e.g. thick borders)
			out[len(out)-1] = (out[len(out)-1] + l) / 2
			continue
		}
		out = append(out, l)
	}
	return out
}

// segmentColumns derives column boundaries from the vertical grid lines and
// returns column crops. Falls back to a proportional grid when detection
// yields too few columns.
func (p *Processor) segmentColumns(img gocv.Mat, grid GridLines) []Column {
	var cols []Column
	xs := grid.VerticalX
	if len(xs) >= 3 {
		// outer pair = table border, inner lines = separators
		x0, x1 := xs[0], xs[len(xs)-1]
		borders := xs[1 : len(xs)-1]
		if len(borders) >= 2 {
			lefts := append([]int{x0}, borders...)
			rights := append(borders, x1)
			for i := 0; i < len(lefts); i++ {
				pad := 2
				cols = append(cols, p.makeColumn(img, i, lefts[i]+pad, rights[i]-pad, grid.Rect))
			}
		}
	}
	if len(cols) < 2 {
		// proportional fallback across the document rect
		rect := grid.Rect
		if rect.Dx() < 50 || rect.Dy() < 50 {
			rect = image.Rect(0, 0, img.Cols(), img.Rows())
		}
		n := 4
		if rect.Dx() < 1000 {
			n = 3
		}
		colW := rect.Dx() / n
		for i := 0; i < n; i++ {
			x0 := rect.Min.X + i*colW
			x1 := x0 + colW
			if i == n-1 {
				x1 = rect.Max.X
			}
			cols = append(cols, p.makeColumn(img, i, x0, x1, grid.Rect))
		}
	}
	return cols
}

// makeColumn crops column i and records its rect (bounded by the table
// rectangle so the title row and footer are excluded).
func (p *Processor) makeColumn(img gocv.Mat, index, x0, x1 int, table image.Rectangle) Column {
	if x0 < 0 {
		x0 = 0
	}
	if x1 > img.Cols() {
		x1 = img.Cols()
	}
	if x1 <= x0 {
		x1 = x0 + 1
	}
	y0, y1 := 0, img.Rows()
	if table.Dy() > 50 {
		y0 = table.Min.Y
		y1 = table.Max.Y
	}
	return Column{
		Index: index,
		X0:    x0,
		X1:    x1,
		Width: x1 - x0,
		Rect:  image.Rect(x0, y0, x1, y1),
	}
}

// buildTextVariant creates the text-optimized column crop: grayscale with
// horizontal grid lines removed. Keeping antialiasing is important for the
// recognizer; thresholding is used only to find the line mask.
func (p *Processor) buildTextVariant(img gocv.Mat, rect image.Rectangle, index int, debug map[string]gocv.Mat) gocv.Mat {
	_ = index
	roi := img.Region(rect)
	defer roi.Close()
	gray := p.grayscale(roi)
	defer gray.Close()

	text := gray.Clone()
	if p.Config.RemoveGridLines {
		bw := gocv.NewMat()
		gocv.Threshold(gray, &bw, 0, 255, gocv.ThresholdBinaryInv+gocv.ThresholdOtsu)
		defer bw.Close()
		// remove horizontal lines: long horizontal runs
		minLen := int(float64(roi.Cols()) * 0.4)
		if minLen < 20 {
			minLen = 20
		}
		hKernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(minLen, 1))
		hLines := gocv.NewMat()
		gocv.MorphologyEx(bw, &hLines, gocv.MorphOpen, hKernel)
		hKernel.Close()
		defer hLines.Close()
		clean := gocv.NewMat()
		gocv.BitwiseOr(gray, hLines, &clean)
		text.Close()
		text = clean
		if debug != nil {
			debug["grid-horizontal.jpg"] = hLines.Clone()
		}
	}

	return text
}
