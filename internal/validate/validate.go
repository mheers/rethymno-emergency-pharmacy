// Package validate performs semantic validation of parsed pharmacy entries
// and cross-checks them against the reference pharmacy catalog from the
// municipality of Rethymno (testdata/reference/rethymno_pharmacies.json).
package validate

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/mheers/rethymno-emergency-pharmacy/internal/normalize"
)

// ReferenceJSON is the embedded municipality pharmacy catalog.
//
//go:embed reference/rethymno_pharmacies.json
var ReferenceJSON []byte

// Reference is one catalog entry.
type Reference struct {
	Name       string  `json:"name"`
	Address    string  `json:"address"`
	Phone      string  `json:"phone"`
	NameLat    string  `json:"name_latin,omitempty"`
	AddressLat string  `json:"address_latin,omitempty"`
	Lat        float64 `json:"lat,omitempty"`
	Lon        float64 `json:"lon,omitempty"`

	// Google carries the optional Places API enrichment (corrected
	// coordinates, contact details, opening hours, photos). It is absent
	// for pharmacies without a Google listing.
	Google *GoogleDetails `json:"google,omitempty"`
}

// GoogleDetails is the Places API enrichment attached to a catalog entry.
type GoogleDetails struct {
	PlaceID             string        `json:"place_id,omitempty"`
	FormattedAddress    string        `json:"formatted_address,omitempty"`
	PhoneInternational  string        `json:"phone_international,omitempty"`
	Website             string        `json:"website,omitempty"`
	GoogleMapsURL       string        `json:"google_maps_url,omitempty"`
	Rating              float64       `json:"rating,omitempty"`
	UserRatingCount     int64         `json:"user_rating_count,omitempty"`
	BusinessStatus      string        `json:"business_status,omitempty"`
	Types               []string      `json:"types,omitempty"`
	OpeningHours        *OpenHours    `json:"opening_hours,omitempty"`
	CurrentOpeningHours *OpenHours    `json:"current_opening_hours,omitempty"`
	Photos              []PhotoDetail `json:"photos,omitempty"`
}

// OpenHours mirrors the Google Places opening-hours structure: human-readable
// weekday descriptions plus the structured weekly periods.
type OpenHours struct {
	WeekdayDescriptions []string `json:"weekday_descriptions,omitempty"`
	Periods             []Period `json:"periods,omitempty"`
	OpenNow             bool     `json:"open_now,omitempty"`
}

// Period is a single open/close interval. Day uses Google numbering where 0
// is Sunday; raw values are preserved.
type Period struct {
	Open  *PeriodPoint `json:"open"`
	Close *PeriodPoint `json:"close,omitempty"`
}

// PeriodPoint is a day/time marker inside an opening-hours period.
type PeriodPoint struct {
	Day    int64 `json:"day"`
	Hour   int64 `json:"hour"`
	Minute int64 `json:"minute"`
}

// PhotoDetail is one downloaded thumbnail embedded in the catalog.
type PhotoDetail struct {
	ContentType string `json:"content_type,omitempty"`
	Base64      string `json:"base64,omitempty"`
}

// Validation is the outcome for one pharmacy.
type Validation struct {
	PhoneValid   bool       `json:"phone_valid"`
	TimeValid    bool       `json:"time_valid"`
	NameGreek    bool       `json:"name_greek"`
	AddressOK    bool       `json:"address_ok"`
	CatalogMatch *Reference `json:"catalog_match,omitempty"`
	Discrepancy  string     `json:"discrepancy,omitempty"`
	Warnings     []string   `json:"warnings,omitempty"`
	Score        float32    `json:"score"`
}

var (
	digitsRe      = regexp.MustCompile(`\d`)
	phone10Re     = regexp.MustCompile(`^2\d{9}$`)
	timeRe        = regexp.MustCompile(`^([01]\d|2[0-3]):[0-5]\d$`)
	streetNumRe   = regexp.MustCompile(`\d{1,4}`)
	greekLetterRe = regexp.MustCompile(`[Α-Ωα-ωάέήίόύώ]`)
)

// ValidatePhone checks the 10-digit landline format.
func ValidatePhone(p string) bool { return phone10Re.MatchString(p) }

// ValidateTime checks HH:MM validity.
func ValidateTime(t string) bool { return timeRe.MatchString(t) }

// ValidateNameGreek requires at least two Greek letters and no digits.
func ValidateNameGreek(name string) bool {
	if len([]rune(name)) < 3 {
		return false
	}
	if len(greekLetterRe.FindAllString(name, -1)) < 2 {
		return false
	}
	if digitsRe.MatchString(name) {
		return false
	}
	return true
}

// ValidateAddress requires a house number or a known street marker.
func ValidateAddress(addr string) bool {
	up := strings.ToUpper(addr)
	if streetNumRe.MatchString(up) {
		return true
	}
	for _, kw := range []string{"ΟΔΟΣ", "ΣΤΡ", "ΛΕΩΦ", "ΠΛΑΤΕΙΑ", "ΠΛ.", "ΟΔ", "STR", "STREET"} {
		if strings.Contains(up, kw) {
			return true
		}
	}
	return false
}

// Validator cross-checks entries against the catalog.
type Validator struct {
	ByPhone   map[string][]Reference
	ByName    map[string]Reference
	MaxOffset int
}

// NewValidator builds a Validator from catalog entries.
func NewValidator(refs []Reference) *Validator {
	v := &Validator{ByPhone: map[string][]Reference{}, ByName: map[string]Reference{}}
	for _, r := range refs {
		for _, digits := range normalize.Phones(r.Phone) {
			v.ByPhone[digits] = append(v.ByPhone[digits], r)
		}
		key := catalogNameKey(r.Name)
		if key != "" {
			if _, ok := v.ByName[key]; !ok {
				v.ByName[key] = r
			}
		}
	}
	return v
}

// LoadReference loads catalog entries from JSON.
func LoadReference(data []byte) ([]Reference, error) {
	var refs []Reference
	if err := json.Unmarshal(data, &refs); err != nil {
		return nil, fmt.Errorf("validate: parse reference: %w", err)
	}
	return refs, nil
}

// catalogNameKey builds a comparison key for a pharmacy name.
func catalogNameKey(name string) string {
	s := strings.ToUpper(normalize.NormalizeGreek(name))
	s = strings.ReplaceAll(s, " ", "")
	s = strings.ReplaceAll(s, "-", "")
	s = strings.ReplaceAll(s, ".", "")
	return s
}

// ValidatePharmacy checks one pharmacy and looks it up in the catalog.
func (v *Validator) ValidatePharmacy(name, address, phone string) Validation {
	return v.ValidatePharmacyWithReference(name, address, phone, nil)
}

// ValidatePharmacyWithReference is ValidatePharmacy with a caller-supplied
// catalog match. The runtime passes the entry an identity adjudicator selected
// for a shared-phone reading, so validation cross-checks that entry instead of
// re-running the similarity pick (TYPESAFE_EVALUATION.md §3A).
func (v *Validator) ValidatePharmacyWithReference(name, address, phone string, preferred *Reference) Validation {
	var res Validation
	res.PhoneValid = ValidatePhone(phone)
	res.TimeValid = true // per-shift check happens elsewhere
	res.NameGreek = ValidateNameGreek(name)
	res.AddressOK = ValidateAddress(address)

	// catalog lookup by phone (digits only, matching NewValidator's keys)
	if candidates := v.ByPhone[normalize.DigitsOnly(phone)]; len(candidates) > 0 {
		ref := ChoosePhoneReference(name, address, candidates)
		if preferred != nil {
			ref = *preferred
		}
		res.CatalogMatch = &ref
		sim := normalize.Similarity(normalize.GreekToLatin(name), normalize.GreekToLatin(ref.Name))
		if sim < 0.45 && name != "" {
			res.Discrepancy = fmt.Sprintf("name mismatch with catalog (%s vs %s)", name, ref.Name)
		}
	} else if phone != "" {
		// phone valid but unknown: search by name similarity
		key := catalogNameKey(name)
		if key != "" {
			bestRef, bestSim := (*Reference)(nil), 0.0
			for _, ref := range v.ByName {
				s := nameSimilarity(key, ref)
				// deterministic tie-break: map iteration order is random
				if s > bestSim || (s == bestSim && bestRef != nil && ref.Name < bestRef.Name) {
					bestSim = s
					bestRef = &ref
				}
			}
			if bestRef != nil && bestSim >= 0.55 {
				res.CatalogMatch = bestRef
				res.Discrepancy = fmt.Sprintf("phone not found in catalog for %s (closest: %s %s)", name, bestRef.Name, bestRef.Phone)
			} else if bestRef != nil && bestSim >= 0.3 {
				res.Discrepancy = fmt.Sprintf("no catalog match for %s (closest: %s)", name, bestRef.Name)
			}
		}
	}

	// warnings
	if !res.PhoneValid && phone != "" {
		res.Warnings = append(res.Warnings, "phone does not match Greek landline format")
	}
	if !res.NameGreek && name != "" {
		res.Warnings = append(res.Warnings, "name lacks plausible Greek characters")
	}
	if !res.AddressOK && address != "" {
		res.Warnings = append(res.Warnings, "address lacks street number or known street pattern")
	}

	// score: start from field validity
	score := float32(0.0)
	if res.PhoneValid {
		score += 0.35
	}
	if res.NameGreek {
		score += 0.3
	}
	if res.AddressOK {
		score += 0.2
	}
	if res.TimeValid {
		score += 0.05
	}
	if res.CatalogMatch != nil && res.Discrepancy == "" {
		score += 0.1
	}
	res.Score = score
	return res
}

// ChoosePhoneReference is the deterministic selection among catalog entries
// that share a phone number: the name similarity against the catalog name,
// raised by the address similarity, with a name-based tie-break. It is
// exported so that callers comparing a System One judgment against today's
// behavior (TYPESAFE_EVALUATION.md §3A) use the same decision the validator
// takes.
func ChoosePhoneReference(name, address string, refs []Reference) Reference {
	if len(refs) == 1 {
		return refs[0]
	}
	best := refs[0]
	bestScore := -1.0
	for _, ref := range refs {
		score := nameSimilarity(catalogNameKey(name), ref)
		addressScore := normalize.Similarity(
			normalize.GreekToLatin(address),
			normalize.GreekToLatin(ref.Address),
		)
		if addressScore > score {
			score = addressScore
		}
		if score > bestScore || (score == bestScore && ref.Name < best.Name) {
			best, bestScore = ref, score
		}
	}
	return best
}

// nameSimilarity compares OCR name against a catalog reference using
// Greeklish-transliterated similarity.
func nameSimilarity(key string, ref Reference) float64 {
	k := catalogNameKey(ref.Name)
	if k == "" {
		return 0
	}
	a := normalize.GreekToLatin(key)
	b := normalize.GreekToLatin(k)
	s := normalize.Similarity(a, b)
	// also compare raw (name may already be Latin from OCR)
	if s2 := normalize.Similarity(key, k); s2 > s {
		s = s2
	}
	return s
}
