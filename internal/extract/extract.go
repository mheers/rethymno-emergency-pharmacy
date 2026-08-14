// Package extract parses the FSKriti HTML page for schedule images and the
// municipality pages for structured duty data and the pharmacy catalog.
package extract

import (
	"fmt"
	"html"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/mheers/rethymno-emergency-pharmacy/internal/normalize"
)

// FSKritiPageURL is the Rethymno duty-schedule page.
const FSKritiPageURL = "https://fskriti.gr/%CE%B5%CF%86%CE%B7%CE%BC%CE%B5%CF%81%CE%AF%CE%B5%CF%82-%CF%86%CE%B1%CF%81%CE%BC%CE%B1%CE%BA%CE%B5%CE%AF%CF%89%CE%BD-%CF%81%CE%B5%CE%B8%CF%8D%CE%BC%CE%BD%CE%BF%CF%85/"

// RethymnoDutyURL is the municipality's structured duty page.
const RethymnoDutyURL = "https://www.rethymno.gr/information-services/pharmacies/pharmacies.html"

// RethymnoCatalogURL is the municipality's full pharmacy list.
const RethymnoCatalogURL = "https://www.rethymno.gr/guide/pharmacies"

var (
	// wp image URLs contain dates like 10.08.2026-17.08.2026_page-0001-2.jpg
	imageURLRe = regexp.MustCompile(`https?://[^"'\s]+?/wp-content/uploads/\d{4}/\d{2}/(\d{2})\.(\d{2})\.(\d{4})-(\d{2})\.(\d{2})\.(\d{4})[^"'\s]*?\.(?:jpg|jpeg|png|webp)`)
	// skip wordpress generated thumbnails (sizes like -1024x724)
	thumbnailRe = regexp.MustCompile(`-\d{2,4}x\d{2,4}\.(jpg|jpeg|png|webp)$`)
)

// ScheduleImage is one weekly schedule image found on the page.
type ScheduleImage struct {
	URL      string
	From, To time.Time
	FullSize bool // true when not a WordPress thumbnail variant
}

// FindScheduleImages extracts all full-size schedule images from the
// FSKriti page HTML, sorted by date.
func FindScheduleImages(pageHTML []byte) []ScheduleImage {
	htmlStr := string(pageHTML)
	var out []ScheduleImage
	seen := map[string]bool{}
	for _, m := range imageURLRe.FindAllStringSubmatch(htmlStr, -1) {
		url := m[0]
		if thumbnailRe.MatchString(url) {
			continue
		}
		if seen[url] {
			continue
		}
		seen[url] = true
		from := mkTime(m[1], m[2], m[3])
		to := mkTime(m[4], m[5], m[6])
		out = append(out, ScheduleImage{URL: url, From: from, To: to, FullSize: true})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].From.Before(out[j].From) })
	return out
}

func mkTime(d, mo, y string) time.Time {
	day := atoi(d)
	month := atoi(mo)
	year := atoi(y)
	return time.Date(year, time.Month(month), day, 0, 0, 0, 0, time.Local)
}

func atoi(s string) int {
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0
		}
		n = n*10 + int(c-'0')
	}
	return n
}

// SelectCurrent picks the schedule image covering today; falls back to the
// latest one.
func SelectCurrent(imgs []ScheduleImage, now time.Time) (ScheduleImage, error) {
	if len(imgs) == 0 {
		return ScheduleImage{}, fmt.Errorf("extract: no schedule images found on page")
	}
	for _, im := range imgs {
		if !now.Before(im.From) && !now.After(im.To) {
			return im, nil
		}
	}
	// fall back to the newest image
	return imgs[len(imgs)-1], nil
}

// MunicipalPharmacy is one entry from the municipality duty page.
type MunicipalPharmacy struct {
	Name    string `json:"name"`
	Address string `json:"address"`
	Phone   string `json:"phone"`
}

// MunicipalShift is one shift block on the municipality duty page.
type MunicipalShift struct {
	From       string              `json:"from"`
	To         string              `json:"to"`
	Pharmacies []MunicipalPharmacy `json:"pharmacies"`
}

// MunicipalDay is one day on the municipality duty page.
type MunicipalDay struct {
	Date   string           `json:"date"` // DD/MM/YYYY
	Day    string           `json:"day"`  // Greek day name
	Shifts []MunicipalShift `json:"shifts"`
}

var (
	dutyDateRe  = regexp.MustCompile(`<h3[^>]*class="duty-date"[^>]*>([^<]+)</h3>`)
	dutyTimeRe  = regexp.MustCompile(`<tr[^>]*class="duty-time"[^>]*><td[^>]*>(.*?)</td></tr>`)
	dutyEntryRe = regexp.MustCompile(`<tr(?:[^>]*)><td[^>]*class="pharmacy-title"[^>]*>(.*?)</td>\s*<td[^>]*class="pharmacy-address"[^>]*>(.*?)</td></tr>`)
	timeWordsRe = regexp.MustCompile(`(\d{1,2}):(\d{2}).*?[\-–—].*?(\d{1,2}):(\d{2})`)
)

// ParseDutyPage parses the municipality duty page into structured days.
func ParseDutyPage(pageHTML []byte) []MunicipalDay {
	s := string(pageHTML)
	var days []MunicipalDay

	// split into day sections
	dateIdx := dutyDateRe.FindAllStringSubmatchIndex(s, -1)
	for i, m := range dateIdx {
		dateText := stripTags(s[m[2]:m[3]])
		d, mo, y, dayName, ok := parseGreekDate(dateText)
		if !ok {
			continue
		}
		secStart := m[1]
		secEnd := len(s)
		if i+1 < len(dateIdx) {
			secEnd = dateIdx[i+1][0]
		}
		day := MunicipalDay{Date: fmt.Sprintf("%02d/%02d/%d", d, mo, y), Day: dayName}
		parseDutySection(s[secStart:secEnd], &day)
		days = append(days, day)
	}
	return days
}

func parseDutySection(section string, day *MunicipalDay) {
	// find time markers and the entries between them
	type marker struct {
		idx  int
		from string
		to   string
	}
	var markers []marker
	for _, m := range dutyTimeRe.FindAllStringSubmatchIndex(section, -1) {
		txt := stripTags(section[m[2]:m[3]])
		tm := timeWordsRe.FindStringSubmatch(txt)
		if tm == nil {
			continue
		}
		from := pad2(tm[1]) + ":" + pad2(tm[2])
		to := pad2(tm[3]) + ":" + pad2(tm[4])
		markers = append(markers, marker{idx: m[0], from: from, to: to})
	}
	if len(markers) == 0 {
		return
	}
	for i, mk := range markers {
		start := mk.idx
		end := len(section)
		if i+1 < len(markers) {
			end = markers[i+1].idx
		}
		shift := MunicipalShift{From: mk.from, To: mk.to}
		for _, em := range dutyEntryRe.FindAllStringSubmatch(section[start:end], -1) {
			name := html.UnescapeString(stripTags(em[1]))
			addrPhone := html.UnescapeString(stripTags(em[2]))
			phone := extractPhone(addrPhone)
			addr := strings.TrimSpace(strings.ReplaceAll(addrPhone, phone, ""))
			addr = strings.Trim(addr, " ,")
			shift.Pharmacies = append(shift.Pharmacies, MunicipalPharmacy{
				Name:    strings.TrimSpace(name),
				Address: addr,
				Phone:   phone,
			})
		}
		day.Shifts = append(day.Shifts, shift)
	}
}

var phoneRe = regexp.MustCompile(`\d{10,11}`)

func extractPhone(s string) string {
	m := phoneRe.FindString(strings.ReplaceAll(s, " ", ""))
	if m == "" {
		return ""
	}
	// keep last 10 digits (drop an 11-digit prefix code if present)
	if len(m) > 10 {
		m = m[len(m)-10:]
	}
	return m
}

var (
	greekMonthRe = map[string]time.Month{
		"ΙΑΝΟΥΑΡΙΟΥ": time.January, "ΦΕΒΡΟΥΑΡΙΟΥ": time.February, "ΜΑΡΤΙΟΥ": time.March,
		"ΑΠΡΙΛΙΟΥ": time.April, "ΜΑΪΟΥ": time.May, "ΜΑΙΟΥ": time.May, "ΙΟΥΝΙΟΥ": time.June,
		"ΙΟΥΛΙΟΥ": time.July, "ΑΥΓΟΥΣΤΟΥ": time.August, "ΣΕΠΤΕΜΒΡΙΟΥ": time.September,
		"ΟΚΤΩΒΡΙΟΥ": time.October, "ΝΟΕΜΒΡΙΟΥ": time.November, "ΔΕΚΕΜΒΡΙΟΥ": time.December,
	}
	greekDayRe = map[string]string{
		"ΚΥΡΙΑΚΗ": "ΚΥΡΙΑΚΗ", "ΔΕΥΤΕΡΑ": "ΔΕΥΤΕΡΑ", "ΤΡΙΤΗ": "ΤΡΙΤΗ", "ΤΕΤΑΡΤΗ": "ΤΕΤΑΡΤΗ",
		"ΠΕΜΠΤΗ": "ΠΕΜΠΤΗ", "ΠΑΡΑΣΚΕΥΗ": "ΠΑΡΑΣΚΕΥΗ", "ΣΑΒΒΑΤΟ": "ΣΑΒΒΑΤΟ",
	}
	dayWordRe = regexp.MustCompile(`(?i)(κυριακ[ηή]|δευτερ[αά]|τριτ[ηή]|τεταρτ[ηή]|πεμπτ[ηή]|παρασκευ[ηή]|σαββατ[οό])`)
)

// parseGreekDate parses "Τετάρτη 12 Αυγούστου 2026".
func parseGreekDate(s string) (d, mo, y int, dayName string, ok bool) {
	up := strings.ToUpper(normalize.NormalizeGreek(stripTags(s)))
	fields := strings.Fields(up)
	if len(fields) < 3 {
		return 0, 0, 0, "", false
	}
	dayName = ""
	if dm := dayWordRe.FindString(up); dm != "" {
		dayName = greekDayRe[strings.ToUpper(dm)]
	}
	// find the number, month name, year
	for i, f := range fields {
		if isDigits(f) {
			d = atoi(f)
			for j := i + 1; j < len(fields); j++ {
				if mon, ok2 := greekMonthRe[strings.ToUpper(fields[j])]; ok2 {
					mo = int(mon)
					for k := j + 1; k < len(fields); k++ {
						if isDigits(fields[k]) {
							y = atoi(fields[k])
							break
						}
					}
					break
				}
			}
			break
		}
	}
	if d == 0 || mo == 0 || y == 0 {
		return 0, 0, 0, "", false
	}
	return d, mo, y, dayName, true
}

func isDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}

func pad2(s string) string {
	if len(s) == 1 {
		return "0" + s
	}
	return s
}

var tagRe = regexp.MustCompile(`<[^>]+>`)

func stripTags(s string) string {
	return html.UnescapeString(strings.TrimSpace(tagRe.ReplaceAllString(s, " ")))
}

// CatalogPharmacy is one row of the municipality pharmacy catalog.
type CatalogPharmacy struct {
	Name    string `json:"name"`
	Address string `json:"address"`
	Phone   string `json:"phone"`
}

var (
	catalogEntryRe = regexp.MustCompile(`<h[34][^>]*>([^<]+)</h[34]>|<p[^>]*>\s*Τηλέφωνο:\s*</p>\s*<p[^>]*>\s*(\d[\d\s\-]*)\s*</p>`)
)

// ParseCatalog extracts pharmacy entries from the catalog page. The page
// format: "Name | Address" heading followed by "Τηλέφωνο:" + number.
func ParseCatalog(pageHTML []byte) []CatalogPharmacy {
	s := string(pageHTML)
	// split into blocks by phone markers
	type block struct {
		start int
		phone string
	}
	var blocks []block
	for _, m := range regexp.MustCompile(`<p[^>]*>\s*Τηλέφωνο:\s*</p>\s*<p[^>]*>\s*([\d\s\-]+)\s*</p>`).FindAllStringSubmatchIndex(s, -1) {
		blocks = append(blocks, block{start: m[0], phone: strings.ReplaceAll(s[m[2]:m[3]], " ", "")})
	}
	var out []CatalogPharmacy
	for i, b := range blocks {
		start := 0
		if i > 0 {
			start = blocks[i-1].start
		}
		section := s[start:b.start]
		// find "Name | Address" heading
		hm := regexp.MustCompile(`<h[34][^>]*>([^<]*(?:\|[^<]*)?)</h[34]>`).FindAllStringSubmatch(section, -1)
		if len(hm) == 0 {
			continue
		}
		parts := strings.SplitN(stripTags(hm[len(hm)-1][1]), "|", 2)
		name := strings.TrimSpace(parts[0])
		addr := ""
		if len(parts) > 1 {
			addr = strings.TrimSpace(parts[1])
		}
		if name == "" {
			continue
		}
		out = append(out, CatalogPharmacy{Name: name, Address: addr, Phone: b.phone})
	}
	return out
}
