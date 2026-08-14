package ocr

// RecognizeColumn recognizes all text rows in a fixed-layout column without
// running the general-purpose text detector. The schedule images are clean
// rasterized tables, so dark-pixel row projection is both cheaper and more
// stable than DB detection on the whole column.
func (e *Engine) RecognizeColumn(img *Image) ([]OCRLine, error) {
	boxes := textRowBoxes(img)
	if len(boxes) == 0 {
		return nil, nil
	}
	return e.RecognizeBoxes(img, boxes)
}

func textRowBoxes(img *Image) [][4][2]float64 {
	if img == nil || img.W == 0 || img.H == 0 || len(img.Data) < img.W*img.H*3 {
		return nil
	}

	dark := make([]int, img.H)
	left := make([]int, img.H)
	right := make([]int, img.H)
	border := minInt(5, img.W/20)
	for y := 0; y < img.H; y++ {
		left[y] = img.W - border
		for x := border; x < img.W-border; x++ {
			i := (y*img.W + x) * 3
			if img.Data[i] >= 210 && img.Data[i+1] >= 210 && img.Data[i+2] >= 210 {
				continue
			}
			dark[y]++
			if x < left[y] {
				left[y] = x
			}
			if x > right[y] {
				right[y] = x
			}
		}
	}

	var boxes [][4][2]float64
	start, gap := -1, 0
	flush := func(end int) {
		if start < 0 || end <= start {
			return
		}
		minX, maxX := img.W, -1
		for y := start; y < end; y++ {
			if left[y] < minX {
				minX = left[y]
			}
			if right[y] > maxX {
				maxX = right[y]
			}
		}
		if maxX < minX || maxX-minX < 2 {
			return
		}
		// A full-width, very short component is a residual table rule, not text.
		if end-start <= 3 && maxX-minX > img.W*9/10 {
			return
		}
		x0 := maxInt(0, minX-3)
		x1 := minInt(img.W, maxX+4)
		y0 := maxInt(0, start-2)
		y1 := minInt(img.H, end+2)
		boxes = append(boxes, [4][2]float64{
			{float64(x0), float64(y0)},
			{float64(x1), float64(y0)},
			{float64(x1), float64(y1)},
			{float64(x0), float64(y1)},
		})
	}

	for y, count := range dark {
		active := count >= 3
		if active {
			if start < 0 {
				start = y
			}
			gap = 0
			continue
		}
		if start < 0 {
			continue
		}
		gap++
		if gap > 1 {
			flush(y - gap + 1)
			start, gap = -1, 0
		}
	}
	if start >= 0 {
		flush(img.H)
	}
	return boxes
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
