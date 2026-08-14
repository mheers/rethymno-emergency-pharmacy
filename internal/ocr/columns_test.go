package ocr

import "testing"

func TestTextRowBoxes(t *testing.T) {
	img := &Image{W: 40, H: 24, Data: make([]byte, 40*24*3)}
	for i := range img.Data {
		img.Data[i] = 255
	}
	for y := 4; y <= 7; y++ {
		for x := 8; x <= 20; x++ {
			for c := 0; c < 3; c++ {
				img.Data[(y*img.W+x)*3+c] = 0
			}
		}
	}
	for y := 13; y <= 16; y++ {
		for x := 22; x <= 35; x++ {
			for c := 0; c < 3; c++ {
				img.Data[(y*img.W+x)*3+c] = 0
			}
		}
	}

	boxes := textRowBoxes(img)
	if len(boxes) != 2 {
		t.Fatalf("got %d row boxes, want 2", len(boxes))
	}
	if got := int(boxes[0][0][1]); got != 2 {
		t.Fatalf("first row top = %d, want 2", got)
	}
	if got := int(boxes[1][0][1]); got != 11 {
		t.Fatalf("second row top = %d, want 11", got)
	}
}
