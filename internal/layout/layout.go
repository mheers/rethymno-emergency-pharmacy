// Package layout reconstructs the schedule table structure from OCR lines
// using spatial relationships: lines are assigned to grid columns, grouped
// into day blocks by date-line anchors, and into shift blocks by time
// markers. No reliance on OCR reading order.
package layout

import (
	"image"
	"image/color"
	"regexp"
	"sort"
	"strings"
	"time"

	"gocv.io/x/gocv"
)

// Line is one OCR result with spatial information.
type Line struct {
	Text       string // normalized text (Greek-canonical)
	RawText    string // original OCR output, never modified
	Confidence float32
	Box        image.Rectangle
}

// DayBlock groups all lines of one day inside a column.
type DayBlock struct {
	HeaderLines []Line // day name / date lines at the top of the day
	Date        time.Time // canonical date parsed from the header, zero when not found
	DayName     string    // expected Greek day name derived from Date
	Lines       []Line
	ShiftMarkers []ShiftMarker
}

// ShiftMarker is a detected "HH:MM-HH:MM" marker line.
type ShiftMarker struct {
	Line Line
	From string
	To   string
}

// ColumnLayout is one grid column with its day blocks.
type ColumnLayout struct {
	Index int
	Rect  image.Rectangle
	Days  []DayBlock
}

// Document is the reconstructed table.
type Document struct {
	TitleLines []Line
	FooterLines []Line
	Columns    []ColumnLayout
	AllLines   []Line
}

var (
	dateRe   = regexp.MustCompile(`^(\d{1,2})[/.](\d{1,2})[/.](\d{4})$`)
	timeRe   = regexp.MustCompile(`^(\d{1,2}):(\d{2})\s*[-–—]\s*(\d{1,2}):(\d{2})$`)
	timeAnyRe = regexp.MustCompile(`(\d{1,2}):(\d{2})\s*[-–—]\s*(\d{1,2}):(\d{2})`)
)

// GreekDayNames maps weekday index (Sunday=0) to the Greek name.
var GreekDayNames = []string{
	"ΚΥΡΙΑΚΗ", "ΔΕΥΤΕΡΑ", "ΤΡΙΤΗ", "ΤΕΤΑΡΤΗ", "ΠΕΜΠΤΗ", "ΠΑΡΑΣΚΕΥΗ", "ΣΑΒΒΑΤΟ",
}

// GreekDayNamesAccent variants for fuzzy day-name matching.
var dayNameVariants = map[string]string{
	"ΔΕΥΤΕΡΑ": "ΔΕΥΤΕΡΑ", "ΤΡΙΤΗ": "ΤΡΙΤΗ", "ΤΕΤΑΡΤΗ": "ΤΕΤΑΡΤΗ", "ΠΕΜΠΤΗ": "ΠΕΜΠΤΗ",
	"ΠΑΡΑΣΚΕΥΗ": "ΠΑΡΑΣΚΕΥΗ", "ΣΑΒΒΑΤΟ": "ΣΑΒΒΑΤΟ", "ΣΑΒΒΑΤΟΝ": "ΣΑΒΒΑΤΟ", "ΚΥΡΙΑΚΗ": "ΚΥΡΙΑΚΗ",
	"ΔΕΥΤΕΡΑΣ": "ΔΕΥΤΕΡΑ", "ΤΡΙΤΗΣ": "ΤΡΙΤΗ", "ΤΕΤΑΡΤΗΣ": "ΤΕΤΑΡΤΗ", "ΠΕΜΠΤΗΣ": "ΠΕΜΠΤΗ",
	"ΠΑΡΑΣΚΕΥΗΣ": "ΠΑΡΑΣΚΕΥΗ", "ΣΑΒΒΑΤΟΥ": "ΣΑΒΒΑΤΟ", "ΚΥΡΙΑΚΗΣ": "ΚΥΡΙΑΚΗ",
}

// NormalizeLine prepares the text used for layout decisions.
func NormalizeLine(raw string) string {
	s := strings.ToUpper(raw)
	s = strings.TrimSpace(s)
	s = strings.ReplaceAll(s, "ΤΗΛ.", "ΤΗΛ")
	s = strings.ReplaceAll(s, "ΤΗΛ:", "ΤΗΛ")
	s = strings.ReplaceAll(s, "ΤΗΛ/", "ΤΗΛ")
	return s
}

// isDateLine matches "DD/MM/YYYY" or "DD.MM.YYYY".
func isDateLine(s string) bool { return dateRe.MatchString(s) }

// isTimeLine matches "HH:MM-HH:MM" with dash variants.
func isTimeLine(s string) bool { return timeRe.MatchString(s) }

// parseDate extracts (day, month, year) from a date line.
func parseDate(s string) (d, m, y int, ok bool) {
	mm := dateRe.FindStringSubmatch(s)
	if mm == nil {
		return 0, 0, 0, false
	}
	ok = true
	for i, v := range mm[1:] {
		n := 0
		for _, c := range v {
			n = n*10 + int(c-'0')
		}
		switch i {
		case 0:
			d = n
		case 1:
			m = n
		case 2:
			y = n
		}
	}
	return
}

// parseTime extracts (from, to) from a time marker line.
func parseTime(s string) (string, string, bool) {
	mm := timeRe.FindStringSubmatch(s)
	if mm == nil {
		return "", "", false
	}
	from := pad2(mm[1]) + ":" + pad2(mm[2])
	to := pad2(mm[3]) + ":" + pad2(mm[4])
	return from, to, true
}

func pad2(s string) string {
	if len(s) == 1 {
		return "0" + s
	}
	return s
}

// ColumnRect couples a column index with its image-space rectangle.
type ColumnRect struct {
	Index int
	Rect  image.Rectangle
}

// Reconstruct assigns OCR lines to columns and day blocks.
//
//	columns: the vision package's detected columns (rects in image space)
//	lines:   OCR lines (boxes in image space)
func Reconstruct(columns []ColumnRect, lines []Line) Document {
	doc := Document{AllLines: lines}

	// vertical band: title above first grid line, footer below last
	minY, maxY := 1<<30, -1
	for _, c := range columns {
		if c.Rect.Min.Y < minY {
			minY = c.Rect.Min.Y
		}
		if c.Rect.Max.Y > maxY {
			maxY = c.Rect.Max.Y
		}
	}

	// assign lines to columns by horizontal center overlap
	colLines := make([][]Line, len(columns))
	for _, l := range lines {
		cx := (l.Box.Min.X + l.Box.Max.X) / 2
		cy := (l.Box.Min.Y + l.Box.Max.Y) / 2
		if cy < minY-10 {
			doc.TitleLines = append(doc.TitleLines, l)
			continue
		}
		if cy > maxY+10 {
			doc.FooterLines = append(doc.FooterLines, l)
			continue
		}
		assigned := false
		for i, c := range columns {
			if cx >= c.Rect.Min.X-8 && cx <= c.Rect.Max.X+8 {
				colLines[i] = append(colLines[i], l)
				assigned = true
				break
			}
		}
		if !assigned {
			// fall back to nearest column
			best, bestDist := -1, 1<<30
			for i, c := range columns {
				dist := abs(cx-c.Rect.Min.X) + abs(cx-c.Rect.Max.X)
				if dist < bestDist {
					bestDist = dist
					best = i
				}
			}
			if best >= 0 {
				colLines[best] = append(colLines[best], l)
			}
		}
	}

	for i, c := range columns {
		col := ColumnLayout{Index: c.Index, Rect: c.Rect}
		col.Days = splitIntoDays(sortLines(colLines[i]))
		doc.Columns = append(doc.Columns, col)
	}
	return doc
}

func sortLines(lines []Line) []Line {
	sort.SliceStable(lines, func(i, j int) bool {
		a, b := lines[i], lines[j]
		if a.Box.Min.Y != b.Box.Min.Y {
			return a.Box.Min.Y < b.Box.Min.Y
		}
		return a.Box.Min.X < b.Box.Min.X
	})
	return lines
}

// splitIntoDays partitions the column's lines into day groups using the
// date lines as anchors. All lines before the first date belong to the
// first day's header. Lines between two date anchors belong to the day
// of the earlier date.
func splitIntoDays(lines []Line) []DayBlock {
	var days []DayBlock
	var current *DayBlock
	var pendingHeader []Line

	for _, l := range lines {
		s := NormalizeLine(l.Text)
		if isDateLine(s) {
			d, m, y, ok := parseDate(s)
			if ok {
				// start a new day block; any pending header lines (day name
				// above the date) belong to this day
				days = append(days, DayBlock{})
				current = &days[len(days)-1]
				current.Date = time.Date(y, time.Month(m), d, 0, 0, 0, 0, time.UTC)
				current.HeaderLines = append(current.HeaderLines, pendingHeader...)
				current.HeaderLines = append(current.HeaderLines, l)
				pendingHeader = nil
				continue
			}
		}
		if current == nil {
			pendingHeader = append(pendingHeader, l)
			continue
		}
		current.Lines = append(current.Lines, l)
	}
	// a trailing day-name-only block (e.g. the last day header partially
	// cut off) is discarded; nothing useful to attach it to
	_ = pendingHeader

	for i := range days {
		day := &days[i]
		if d, m, y, ok := parseDateFromAny(day.HeaderLines); ok {
			day.Date = time.Date(y, time.Month(m), d, 0, 0, 0, 0, time.UTC)
			day.DayName = greekDayFor(d, m, y)
		}
		// extract shift markers from the day's lines
		var rest []Line
		for _, l := range day.Lines {
			s := NormalizeLine(l.Text)
			if isTimeLine(s) {
				from, to, _ := parseTime(s)
				day.ShiftMarkers = append(day.ShiftMarkers, ShiftMarker{Line: l, From: from, To: to})
			} else {
				rest = append(rest, l)
			}
		}
		day.Lines = rest
	}
	return days
}

func parseDateFromAny(lines []Line) (d, m, y int, ok bool) {
	for _, l := range lines {
		if d, m, y, ok2 := parseDate(NormalizeLine(l.Text)); ok2 {
			return d, m, y, true
		}
	}
	return 0, 0, 0, false
}

// greekDayFor returns the Greek day name for a date (Gregorian, proleptic).
func greekDayFor(d, m, y int) string {
	idx := dayOfWeek(d, m, y)
	if idx < 0 {
		return ""
	}
	return GreekDayNames[idx]
}

// dayOfWeek implements Zeller's congruence (Sunday=0 ... Saturday=6).
func dayOfWeek(d, m, y int) int {
	if m < 3 {
		m += 12
		y--
	}
	k := y % 100
	j := y / 100
	h := (d + 13*(m+1)/5 + k + k/4 + j/4 + 5*j) % 7
	// h: Saturday=0 ... Friday=6  -> convert to Sunday=0
	return (h + 6) % 7
}

// AssignShiftContent partitions each day's lines into shift blocks using
// the shift markers as anchors. Lines before the first marker belong to
// the first shift; lines after the last marker belong to the last shift.
func AssignShiftContent(day *DayBlock) []ShiftBlock {
	markers := day.ShiftMarkers
	if len(markers) == 0 {
		return []ShiftBlock{{From: "", To: "", Lines: day.Lines}}
	}
	sort.SliceStable(markers, func(i, j int) bool {
		return markers[i].Line.Box.Min.Y < markers[j].Line.Box.Min.Y
	})
	lines := day.Lines
	sort.SliceStable(lines, func(i, j int) bool {
		return lines[i].Box.Min.Y < lines[j].Box.Min.Y
	})

	blocks := make([]ShiftBlock, 0, len(markers))
	for mi, mk := range markers {
		block := ShiftBlock{From: mk.From, To: mk.To}
		y0 := mk.Line.Box.Min.Y
		y1 := 1 << 30
		if mi+1 < len(markers) {
			y1 = markers[mi+1].Line.Box.Min.Y
		}
		for _, l := range lines {
			cy := (l.Box.Min.Y + l.Box.Max.Y) / 2
			if cy >= y0 && cy < y1 {
				block.Lines = append(block.Lines, l)
			}
		}
		blocks = append(blocks, block)
	}
	// lines above the first marker
	firstY := markers[0].Line.Box.Min.Y
	var head []Line
	for _, l := range lines {
		if l.Box.Max.Y < firstY {
			head = append(head, l)
		}
	}
	if len(head) > 0 {
		blocks[0].Lines = append(head, blocks[0].Lines...)
		sortLines(blocks[0].Lines)
	}
	return blocks
}

// ShiftBlock is a time-bounded group of pharmacy lines.
type ShiftBlock struct {
	From  string
	To    string
	Lines []Line
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// DrawBoxes annotates an image with OCR boxes (for debug output).
func DrawBoxes(img *gocv.Mat, lines []Line) {
	for _, l := range lines {
		gocv.Rectangle(img, l.Box, color.RGBA{R: 255, B: 0, G: 0, A: 255}, 1)
	}
}
