// Package parse reconstructs pharmacy entries from layout shift blocks:
// name / address / phone fields are extracted with domain knowledge of the
// Rethymno schedule format (Greek name, Greek address, Latin address
// transliteration with "STR." suffix, "ΤΗΛ" phone line).
package parse

import (
	"regexp"
	"strings"
	"time"

	"github.com/mheers/rethymno-emergency-pharmacy/internal/layout"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/normalize"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/validate"
)

// Pharmacy is one reconstructed pharmacy entry.
type Pharmacy struct {
	Name       string   `json:"name"`
	Address    string   `json:"address"`
	Phone      string   `json:"phone"`
	NameLat    string   `json:"name_latin,omitempty"`
	AddressLat string   `json:"address_latin,omitempty"`
	Lat        float64  `json:"lat,omitempty"`
	Lon        float64  `json:"lon,omitempty"`
	Confidence float32  `json:"confidence"`
	Warnings   []string `json:"warnings,omitempty"`

	// Google carries the optional Places API enrichment from the
	// reference catalog (corrected coordinates, contact details,
	// opening hours, photos). It is absent for unmatched pharmacies.
	Google *validate.GoogleDetails `json:"google,omitempty"`
}

// Shift is one duty shift with its pharmacies.
type Shift struct {
	From       string     `json:"from"`
	To         string     `json:"to"`
	Pharmacies []Pharmacy `json:"pharmacies"`
}

// DaySchedule is one calendar day.
type DaySchedule struct {
	Date   time.Time `json:"date"` // canonical (RFC 3339) date
	Day    string    `json:"day"`
	Shifts []Shift   `json:"shifts"`
}

// Schedule is the full parsed schedule document.
type Schedule struct {
	SourceURL string        `json:"source_url"`
	City      string        `json:"city"`
	Days      []DaySchedule `json:"days"`
}

var (
	dateLineRe = regexp.MustCompile(`^\d{1,2}[/.]\d{1,2}[/.]\d{4}$`)

	phonePrefixRe  = regexp.MustCompile(`(?i)(τηλ|thl)`)
	digitsRe       = regexp.MustCompile(`\d`)
	houseNumberRe  = regexp.MustCompile(`\d{1,4}`)
	latinOnlyRe    = regexp.MustCompile(`(?i)[a-z]`)
	greekRe        = regexp.MustCompile(`[Α-Ωα-ωάέήίόύώΐΰ]`)
	streetSuffixRe = regexp.MustCompile(`(?i)(str\.?|street|οδος|οδ|στρ\.?|λεωφ\.?|πλ\.?|πλατει)`)
	stopWords      = map[string]bool{
		"ΤΗΛ": true, "ΤΗΛ.": true, "ΤΗΛ/": true, "THL": true, "THL.": true,
		"ΦΑΡΜΑΚΕΙΟ": true, "ΕΦΗΜΕΡΙΕΣ": true,
	}
	// addressDescriptionPrefixes mark lines that describe a location
	// ("απέναντι από...", "κάτω από...") rather than a pharmacy name.
	addressDescriptionPrefixes = []string{
		"ΕΝΑΝΤΙ", "ΑΠΕΝΑΝΤΙ", "ΚΑΤΩ", "ΠΛΗΣΙΟΝ", "ΑΠΕΝΑΝΤΙΑ", "ΔΙΠΛΑ",
		"ΣΩΜΑΤΕΙΟ", "ΣΩΜΑ", "ΕΝΑΝΤΙΑ", "ΠΕΡΙΟΧΗ",
	}
)

// ParseDocument converts a layout.Document into a Schedule. SourceURL and
// City are injected by the caller.
func ParseDocument(doc *layout.Document, sourceURL, city string) Schedule {
	sched := Schedule{SourceURL: sourceURL, City: city}
	if city == "" {
		sched.City = "Ρέθυμνο"
	}
	for _, col := range doc.Columns {
		for _, day := range col.Days {
			ds := DaySchedule{Date: day.Date, Day: day.DayName}
			for _, sb := range layout.AssignShiftContent(&day) {
				shift := Shift{From: sb.From, To: sb.To}
				shift.Pharmacies = parseShift(sb.Lines)
				if len(shift.Pharmacies) > 0 {
					ds.Shifts = append(ds.Shifts, shift)
				}
			}
			if len(ds.Shifts) > 0 {
				sched.Days = append(sched.Days, ds)
			}
		}
	}
	return sched
}

// parseShift groups the shift's OCR lines into pharmacy entries. The known
// structure per entry is [NAME] [ADDRESS-GREEK] [ADDRESS-LATIN] [ΤΗΛ phone],
// and the phone line terminates an entry.
func parseShift(lines []layout.Line) []Pharmacy {
	if len(lines) == 0 {
		return nil
	}
	lines = sortByY(lines)

	// find phone lines: they close each entry
	phoneIdx := map[int]bool{}
	for i, l := range lines {
		if isPhoneLine(l.Text) {
			phoneIdx[i] = true
		}
	}

	var entries [][]layout.Line
	var cur []layout.Line
	for i, l := range lines {
		cur = append(cur, l)
		if phoneIdx[i] {
			entries = append(entries, cur)
			cur = nil
		}
	}
	if len(cur) > 0 {
		if len(entries) == 0 {
			entries = append(entries, cur)
		} else {
			// trailing lines without phone: merge into the last entry
			entries[len(entries)-1] = append(entries[len(entries)-1], cur...)
		}
	}

	var out []Pharmacy
	for _, e := range entries {
		p := reconstructEntry(e)
		if p.Name != "" || p.Phone != "" {
			out = append(out, p)
		}
	}
	return out
}

// reconstructEntry extracts name/address/phone from one entry's lines.
func reconstructEntry(lines []layout.Line) Pharmacy {
	var p Pharmacy
	var nameLines, addrLines []string

	for _, l := range lines {
		up := strings.ToUpper(strings.TrimSpace(l.Text))
		if isPhoneLine(up) {
			digits := normalize.DigitsOnly(up)
			if len(digits) == 10 {
				p.Phone = digits
			}
			continue
		}
		if isJunkLine(up) {
			continue
		}
		if isTimeOrDateLine(up) || isDateLine(up) || isDayName(up) {
			continue
		}
		if looksLikeAddress(up) {
			addrLines = append(addrLines, cleanField(up))
			continue
		}
		if looksLikeName(up) {
			nameLines = append(nameLines, cleanField(up))
			continue
		}
		// ambiguous line: merge into the current field based on digits
		if countDigits(up) > 0 {
			addrLines = append(addrLines, cleanField(up))
		} else {
			nameLines = append(nameLines, cleanField(up))
		}
	}

	// drop the Latin transliteration line when a Greek address line exists
	p.Address = mergeAddressLines(addrLines)
	p.Name = mergeLines(nameLines)
	return p
}

// mergeAddressLines keeps only the Greek address when both Greek and Latin
// transliterations of the same address are present.
func mergeAddressLines(lines []string) string {
	if len(lines) == 0 {
		return ""
	}
	var greek, latin []string
	for _, l := range lines {
		if greekRe.MatchString(l) {
			greek = append(greek, l)
		} else {
			latin = append(latin, l)
		}
	}
	if len(greek) > 0 {
		return mergeLines(greek)
	}
	return mergeLines(latin)
}

// isPhoneLine detects "ΤΗΛ ..." or a line dominated by digits.
func isPhoneLine(s string) bool {
	up := strings.ToUpper(strings.TrimSpace(s))
	if phonePrefixRe.MatchString(up) {
		digits := normalize.DigitsOnly(up)
		if len(digits) >= 6 {
			return true
		}
	}
	d := countDigits(up)
	return d >= 6 && float64(d)/float64(max(1, lenRunes(up))) > 0.5
}

// looksLikeAddress: contains a house number and/or a street keyword.
func looksLikeAddress(s string) bool {
	if len(s) == 0 {
		return false
	}
	hasNumber := houseNumberRe.MatchString(s)
	hasStreet := streetSuffixRe.MatchString(s)
	if hasStreet {
		return true
	}
	if hasNumber && latinOnlyRe.MatchString(s) && strings.Contains(strings.ToUpper(s), "STR") {
		return true
	}
	if hasNumber && lenRunes(s) > 6 && !greekRe.MatchString(s) {
		return true
	}
	return false
}

// looksLikeName: Greek letters, no digits, not a known stop word and not an
// address-description line.
func looksLikeName(s string) bool {
	if len(s) == 0 {
		return false
	}
	if stopWords[strings.ToUpper(s)] {
		return false
	}
	if countDigits(s) > 0 {
		return false
	}
	if !greekRe.MatchString(s) {
		return false
	}
	if lenRunes(s) < 3 {
		return false
	}
	up := strings.ToUpper(s)
	for _, p := range addressDescriptionPrefixes {
		if strings.HasPrefix(up, p) {
			return false
		}
	}
	return true
}

// isJunkLine drops low-information OCR noise.
func isJunkLine(s string) bool {
	up := strings.TrimSpace(s)
	if up == "" || up == "/" || up == "-" || up == "." {
		return true
	}
	if !greekRe.MatchString(up) && !latinOnlyRe.MatchString(up) && countDigits(up) == 0 {
		return true
	}
	return false
}

func isTimeOrDateLine(s string) bool {
	return strings.Contains(s, ":") && (strings.Contains(s, "-") || strings.Contains(s, "/"))
}

func isDateLine(s string) bool { return dateLineRe.MatchString(s) }

// isDayName skips stray day-header words inside shift content.
func isDayName(s string) bool {
	up := strings.ToUpper(strings.TrimSpace(s))
	for _, d := range []string{"ΔΕΥΤΕΡΑ", "ΤΡΙΤΗ", "ΤΕΤΑΡΤΗ", "ΠΕΜΠΤΗ", "ΠΑΡΑΣΚΕΥΗ", "ΣΑΒΒΑΤΟ", "ΚΥΡΙΑΚΗ"} {
		if up == d {
			return true
		}
	}
	return false
}

// cleanField strips punctuation artifacts and collapses whitespace.
func cleanField(s string) string {
	s = normalize.NormalizeGreek(s)
	up := strings.ToUpper(s)
	up = strings.Trim(up, ".,;-—|/\"' ")
	return strings.TrimSpace(up)
}

func mergeLines(lines []string) string {
	return strings.Join(lines, " ")
}

func sortByY(lines []layout.Line) []layout.Line {
	for i := 1; i < len(lines); i++ {
		for j := i; j > 0 && lines[j].Box.Min.Y < lines[j-1].Box.Min.Y; j-- {
			lines[j], lines[j-1] = lines[j-1], lines[j]
		}
	}
	return lines
}

func countDigits(s string) int {
	n := 0
	for _, r := range s {
		if r >= '0' && r <= '9' {
			n++
		}
	}
	return n
}

func lenRunes(s string) int {
	return len([]rune(s))
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

var _ = digitsRe
