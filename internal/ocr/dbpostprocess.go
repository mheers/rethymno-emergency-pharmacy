package ocr

import (
	"fmt"
	"math"
	"os"
)

// dbPostprocess implements the Differentiable Binarization postprocessor
// in pure Go. It mirrors PaddleOCR's DBPostProcess semantics:
//
//  1. threshold the probability map at DetThresh
//  2. trace connected components (RETR_LIST equivalent)
//  3. bounding box per component, drop boxes with min side < 3
//  4. compute the box score on the ORIGINAL probability map, keep >= box_thresh
//  5. unclip the box by area*unclip/perimeter
//  6. clip to image bounds and scale back to input-image coordinates
//
// For this document family the text lines are axis-aligned, so an
// axis-aligned bounding box replaces cv::minAreaRect; scores are computed
// over the bounding box (slightly stricter than the rotated polygon, which
// additionally rejects grid-line detections).
func dbPostprocess(prob []float32, ph, pw int, cfg Config, scale float64) [][4][2]float64 {
	if ph <= 0 || pw <= 0 {
		return nil
	}
	bitmap := make([]byte, ph*pw)
	for i, v := range prob {
		if v > float32(cfg.DetThresh) {
			bitmap[i] = 1
		}
	}
	if os.Getenv("OCR_DEBUG") != "" {
		var mx, over3 float64
		for _, v := range prob {
			f := float64(v)
			if f > mx {
				mx = f
			}
			if f > 0.3 {
				over3++
			}
		}
		fmt.Printf("dbpost: max=%.4f >0.3=%d/%d\n", mx, int(over3), len(prob))
	}

	var boxes [][4][2]float64
	seen := make([]bool, len(bitmap))
	for y := 0; y < ph; y++ {
		for x := 0; x < pw; x++ {
			i := y*pw + x
			if bitmap[i] == 0 || seen[i] {
				continue
			}
			// flood fill to find the component, collecting its bbox
			minX, minY, maxX, maxY := x, y, x, y
			stack := [][2]int{{x, y}}
			seen[i] = true
			for len(stack) > 0 {
				px, py := stack[len(stack)-1][0], stack[len(stack)-1][1]
				stack = stack[:len(stack)-1]
				for dy := -1; dy <= 1; dy++ {
					for dx := -1; dx <= 1; dx++ {
						nx, ny := px+dx, py+dy
						if nx < 0 || ny < 0 || nx >= pw || ny >= ph {
							continue
						}
						ni := ny*pw + nx
						if bitmap[ni] == 0 || seen[ni] {
							continue
						}
						seen[ni] = true
						if nx < minX {
							minX = nx
						}
						if nx > maxX {
							maxX = nx
						}
						if ny < minY {
							minY = ny
						}
						if ny > maxY {
							maxY = ny
						}
						stack = append(stack, [2]int{nx, ny})
					}
				}
			}
			w := maxX - minX + 1
			h := maxY - minY + 1
			if w < 3 || h < 3 {
				continue
			}
			score := boxScore(prob, ph, pw, minX, minY, maxX, maxY)
			if os.Getenv("OCR_DEBUG") != "" && len(boxes) < 5 {
				fmt.Printf("  comp bbox=(%d,%d,%d,%d) w=%d h=%d score=%.3f\n", minX, minY, maxX, maxY, w, h, score)
			}
			if score < cfg.DetBoxThresh {
				continue
			}
			// unclip: expand the box by dist in all directions
			dist := float64(w*h) * cfg.DetUnclip / (2 * float64(w+h))
			if dist < 1 {
				dist = 1
			}
			ux0 := int(float64(minX) - dist)
			uy0 := int(float64(minY) - dist)
			ux1 := int(float64(maxX) + dist)
			uy1 := int(float64(maxY) + dist)
			if ux1-ux0 < 5 || uy1-uy0 < 5 {
				continue
			}
			// clip to prob-map bounds then scale to input-image coordinates
			cx0 := clampInt(ux0, 0, pw-1)
			cy0 := clampInt(uy0, 0, ph-1)
			cx1 := clampInt(ux1, 0, pw-1)
			cy1 := clampInt(uy1, 0, ph-1)
			boxes = append(boxes, [4][2]float64{
				{float64(cx0) / scale, float64(cy0) / scale},
				{float64(cx1) / scale, float64(cy0) / scale},
				{float64(cx1) / scale, float64(cy1) / scale},
				{float64(cx0) / scale, float64(cy1) / scale},
			})
		}
	}
	return boxes
}

// boxScore computes the mean probability over the axis-aligned box.
func boxScore(prob []float32, ph, pw int, x0, y0, x1, y1 int) float64 {
	var sum float64
	var n int
	for y := y0; y <= y1; y++ {
		row := y * pw
		for x := x0; x <= x1; x++ {
			sum += float64(prob[row+x])
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

var _ = math.Max
