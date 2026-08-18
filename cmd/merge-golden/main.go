// Command merge-golden merges the Google-enriched golden pharmacy catalog
// (produced by the guide-maps enrichment workflow in the expat-map-guide
// repository) into the embedded reference catalog used for OCR validation and
// schedule enrichment.
//
// Usage:
//
//	go run ./cmd/merge-golden \
//	  -golden /path/to/expat-map-guide/catalog/pharmacies.json \
//	  -reference internal/validate/reference/rethymno_pharmacies.json \
//	  -out internal/validate/reference/rethymno_pharmacies.json
//
// Matching is by normalized phone digits. Reference entries without a phone
// match keep their existing OpenStreetMap coordinates and no Google block.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/mheers/rethymno-emergency-pharmacy/internal/normalize"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/validate"
)

// goldenCatalog mirrors the expat-map-guide enrichment output envelope. Only
// the fields needed for the reference merge are modeled.
type goldenCatalog struct {
	Pharmacies []goldenPharmacy `json:"pharmacies"`
}

// goldenPharmacy is one enriched record from the golden catalog.
type goldenPharmacy struct {
	Source struct {
		Name  string `json:"name"`
		Phone string `json:"phone"`
	} `json:"source"`
	Matched    bool          `json:"matched"`
	PlaceID    string        `json:"placeId,omitempty"`
	Name       string        `json:"name,omitempty"`
	Phone      string        `json:"phone,omitempty"`
	PhoneIntl  string        `json:"phoneInternational,omitempty"`
	WebsiteURL string        `json:"websiteUrl,omitempty"`
	MapsURL    string        `json:"googleMapsUrl,omitempty"`
	Formatted  string        `json:"formattedAddress,omitempty"`
	Latitude   float64       `json:"latitude"`
	Longitude  float64       `json:"longitude"`
	TimeZone   string        `json:"timeZone,omitempty"`
	OffsetMin  int64         `json:"utcOffsetMinutes,omitempty"`
	Rating     float64       `json:"rating,omitempty"`
	ReviewCnt  int64         `json:"userRatingCount,omitempty"`
	Status     string        `json:"businessStatus,omitempty"`
	Types      []string      `json:"types,omitempty"`
	Hours      *goldenHours  `json:"openingHours,omitempty"`
	CurHours   *goldenHours  `json:"currentOpeningHours,omitempty"`
	Photos     []goldenPhoto `json:"photos,omitempty"`
}

type goldenHours struct {
	WeekdayDescriptions []string       `json:"weekdayDescriptions,omitempty"`
	Periods             []goldenPeriod `json:"periods,omitempty"`
	OpenNow             bool           `json:"openNow,omitempty"`
}

type goldenPeriod struct {
	Open  *goldenPoint `json:"open"`
	Close *goldenPoint `json:"close,omitempty"`
}

type goldenPoint struct {
	Day    int64 `json:"day"`
	Hour   int64 `json:"hour"`
	Minute int64 `json:"minute"`
}

type goldenPhoto struct {
	ContentType string `json:"contentType,omitempty"`
	Base64      string `json:"base64,omitempty"`
}

func main() {
	var (
		goldenPath string
		refPath    string
		outPath    string
		pretty     bool
	)
	flag.StringVar(&goldenPath, "golden", "", "golden catalog JSON from the enrichment workflow (required)")
	flag.StringVar(&refPath, "reference", "internal/validate/reference/rethymno_pharmacies.json", "reference catalog JSON to merge into")
	flag.StringVar(&outPath, "out", "", "output path (default: overwrite the reference)")
	flag.BoolVar(&pretty, "pretty", true, "indent the output JSON")
	flag.Parse()

	if goldenPath == "" {
		fmt.Fprintln(os.Stderr, "-golden is required")
		os.Exit(2)
	}
	if outPath == "" {
		outPath = refPath
	}

	golden, err := loadGolden(goldenPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "golden: %v\n", err)
		os.Exit(1)
	}
	refs, err := validate.LoadReference(mustRead(refPath))
	if err != nil {
		fmt.Fprintf(os.Stderr, "reference: %v\n", err)
		os.Exit(1)
	}

	byPhone := map[string][]goldenPharmacy{}
	matched := 0
	for _, g := range golden {
		key := normalize.DigitsOnly(firstNonEmpty(g.Phone, g.Source.Phone))
		if key == "" || !g.Matched {
			continue
		}
		byPhone[key] = append(byPhone[key], g)
		matched++
	}

	merged := 0
	skipped := 0
	// Group reference entries by phone so duplicate reference phones (two
	// entries sharing one duty number) can be disambiguated by name. Dual
	// phone fields (e.g. "28310 51113, 28310 55649") register under every
	// individual number.
	byRefPhone := map[string][]int{}
	for i := range refs {
		for _, key := range normalize.Phones(refs[i].Phone) {
			byRefPhone[key] = append(byRefPhone[key], i)
		}
	}
	for phoneKey, indexes := range byRefPhone {
		candidates := byPhone[phoneKey]
		if len(candidates) == 0 {
			continue
		}
		for _, idx := range indexes {
			ref := &refs[idx]
			g, ok := bestGoldenByName(ref.Name, candidates)
			if !ok {
				skipped++
				continue
			}
			if len(indexes) > 1 {
				// Two reference entries share this phone: only the best
				// name match receives the enrichment; the other keeps its
				// OSM data.
				if goldenNameScore(ref.Name, g) < 0.55 {
					skipped++
					continue
				}
			}
			ref.Lat = g.Latitude
			ref.Lon = g.Longitude
			ref.Google = toGoogleDetails(g)
			merged++
		}
	}

	fmt.Printf("golden records: %d (matched: %d)\n", len(golden), matched)
	fmt.Printf("reference entries: %d (merged: %d, ambiguous skipped: %d)\n", len(refs), merged, skipped)

	data, err := json.MarshalIndent(refs, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode: %v\n", err)
		os.Exit(1)
	}
	if pretty {
		data = append(data, '\n')
	}
	if err := os.WriteFile(outPath, data, 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "write %s: %v\n", outPath, err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s\n", outPath)
}

func loadGolden(path string) ([]goldenPharmacy, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var catalog goldenCatalog
	if err := json.Unmarshal(data, &catalog); err != nil {
		return nil, err
	}
	return catalog.Pharmacies, nil
}

// goldenNameScore measures how well a reference name matches a golden record.
func goldenNameScore(refName string, g goldenPharmacy) float64 {
	score := normalize.Similarity(normalize.GreekToLatin(refName), normalize.GreekToLatin(g.Name))
	if s2 := normalize.Similarity(normalize.GreekToLatin(refName), normalize.GreekToLatin(g.Source.Name)); s2 > score {
		score = s2
	}
	return score
}

// bestGoldenByName picks the golden record whose name best matches a reference
// name. When the reference name is unknown or the best match is too weak, it
// reports false so the caller skips the ambiguous entry.
func bestGoldenByName(refName string, candidates []goldenPharmacy) (goldenPharmacy, bool) {
	if len(candidates) == 1 {
		return candidates[0], true
	}
	best := candidates[0]
	bestScore := -1.0
	for _, g := range candidates {
		score := goldenNameScore(refName, g)
		if score > bestScore {
			best, bestScore = g, score
		}
	}
	if bestScore < 0.4 {
		return goldenPharmacy{}, false
	}
	return best, true
}

func toGoogleDetails(g goldenPharmacy) *validate.GoogleDetails {
	detail := &validate.GoogleDetails{
		PlaceID:             g.PlaceID,
		FormattedAddress:    g.Formatted,
		PhoneInternational:  g.PhoneIntl,
		Website:             g.WebsiteURL,
		GoogleMapsURL:       g.MapsURL,
		Rating:              g.Rating,
		UserRatingCount:     g.ReviewCnt,
		BusinessStatus:      g.Status,
		Types:               g.Types,
		OpeningHours:        toOpenHours(g.Hours),
		CurrentOpeningHours: toOpenHours(g.CurHours),
	}
	for _, p := range g.Photos {
		if p.Base64 == "" {
			continue
		}
		detail.Photos = append(detail.Photos, validate.PhotoDetail{ContentType: p.ContentType, Base64: p.Base64})
	}
	if detail.PlaceID == "" && detail.FormattedAddress == "" && len(detail.Photos) == 0 {
		return nil
	}
	return detail
}

func toOpenHours(h *goldenHours) *validate.OpenHours {
	if h == nil {
		return nil
	}
	out := &validate.OpenHours{
		WeekdayDescriptions: h.WeekdayDescriptions,
		OpenNow:             h.OpenNow,
	}
	for _, p := range h.Periods {
		period := validate.Period{}
		if p.Open != nil {
			period.Open = &validate.PeriodPoint{Day: p.Open.Day, Hour: p.Open.Hour, Minute: p.Open.Minute}
		}
		if p.Close != nil {
			period.Close = &validate.PeriodPoint{Day: p.Close.Day, Hour: p.Close.Hour, Minute: p.Close.Minute}
		}
		out.Periods = append(out.Periods, period)
	}
	return out
}

func mustRead(path string) []byte {
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read %s: %v\n", path, err)
		os.Exit(1)
	}
	return data
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
