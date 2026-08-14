package layout

import (
	"image"
	"testing"
)

func mkLine(y int, text string) Line {
	return Line{
		Text:       text,
		RawText:    text,
		Confidence: 0.9,
		Box:        image.Rect(300, y, 500, y+20),
	}
}

func TestReconstructDays(t *testing.T) {
	cols := []ColumnRect{{Index: 0, Rect: image.Rect(271, 106, 572, 1133)}}
	lines := []Line{
		mkLine(160, "ΔΕΥΤΕΡΑ"),
		mkLine(180, "10/08/2026"),
		mkLine(210, "08:00-21:00"),
		mkLine(240, "ΑΛΕΦΑΝΤΙΝΟΥ ΜΑΡΙΑ"),
		mkLine(470, "21:00-08:00"),
		mkLine(630, "ΠΑΡΑΣΚΕΥΗ"),
		mkLine(650, "14/08/2026"),
		mkLine(680, "08:00-21:00"),
	}
	doc := Reconstruct(cols, lines)
	if len(doc.Columns) != 1 {
		t.Fatalf("expected 1 column, got %d", len(doc.Columns))
	}
	days := doc.Columns[0].Days
	if len(days) != 2 {
		t.Fatalf("expected 2 days, got %d", len(days))
	}
	if days[0].Date.Format("02/01/2006") != "10/08/2026" || days[0].DayName != "ΔΕΥΤΕΡΑ" {
		t.Errorf("day 0 wrong: %+v", days[0])
	}
	if days[1].Date.Format("02/01/2006") != "14/08/2026" || days[1].DayName != "ΠΑΡΑΣΚΕΥΗ" {
		t.Errorf("day 1 wrong: %+v", days[1])
	}
	if len(days[0].ShiftMarkers) != 2 {
		t.Errorf("day 0 should have 2 shift markers, got %d", len(days[0].ShiftMarkers))
	}
	m0 := days[0].ShiftMarkers[0]
	if m0.From != "08:00" || m0.To != "21:00" {
		t.Errorf("shift 0 wrong: %+v", m0)
	}
	m1 := days[0].ShiftMarkers[1]
	if m1.From != "21:00" || m1.To != "08:00" {
		t.Errorf("shift 1 wrong: %+v", m1)
	}
}

func TestAssignShiftContent(t *testing.T) {
	day := DayBlock{
		ShiftMarkers: []ShiftMarker{
			{Line: mkLine(210, "08:00-21:00"), From: "08:00", To: "21:00"},
			{Line: mkLine(470, "21:00-08:00"), From: "21:00", To: "08:00"},
		},
		Lines: []Line{
			mkLine(240, "ΑΛΕΦΑΝΤΙΝΟΥ ΜΑΡΙΑ"),
			mkLine(500, "ΔΑΦΝΟΜΗΛΗ ΓΕΩΡΓΙΑ"),
		},
	}
	blocks := AssignShiftContent(&day)
	if len(blocks) != 2 {
		t.Fatalf("expected 2 blocks, got %d", len(blocks))
	}
	if len(blocks[0].Lines) != 1 || blocks[0].Lines[0].Text != "ΑΛΕΦΑΝΤΙΝΟΥ ΜΑΡΙΑ" {
		t.Errorf("block 0 wrong: %+v", blocks[0].Lines)
	}
	if len(blocks[1].Lines) != 1 || blocks[1].Lines[0].Text != "ΔΑΦΝΟΜΗΛΗ ΓΕΩΡΓΙΑ" {
		t.Errorf("block 1 wrong: %+v", blocks[1].Lines)
	}
}

func TestDayOfWeek(t *testing.T) {
	cases := []struct {
		d, m, y int
		want    string
	}{
		{10, 8, 2026, "ΔΕΥΤΕΡΑ"},
		{11, 8, 2026, "ΤΡΙΤΗ"},
		{12, 8, 2026, "ΤΕΤΑΡΤΗ"},
		{13, 8, 2026, "ΠΕΜΠΤΗ"},
		{14, 8, 2026, "ΠΑΡΑΣΚΕΥΗ"},
		{15, 8, 2026, "ΣΑΒΒΑΤΟ"},
		{16, 8, 2026, "ΚΥΡΙΑΚΗ"},
	}
	for _, c := range cases {
		if got := greekDayFor(c.d, c.m, c.y); got != c.want {
			t.Errorf("greekDayFor(%d/%d/%d) = %s, want %s", c.d, c.m, c.y, got, c.want)
		}
	}
}
