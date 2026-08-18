// Package pipeline_test contains golden tests that exercise the full OCR
// pipeline against the real schedule images. These tests require the ONNX
// models (./models) and must run in the Docker build environment:
//
//	docker build -t rethymno-emergency-pharmacy:dev .
//	docker run --rm -v $PWD:/data -w /data rethymno-emergency-pharmacy:dev go test ./internal/pipeline/ -run Golden -v
//
// The OCR worker runs in a dedicated subprocess (see internal/ocr/worker.go).
// TestMain builds the real rethymno-emergency-pharmacy binary for that purpose: spawning
// the test binary itself does NOT work, because a test binary ignores
// positional arguments and re-runs all tests, spawning an unbounded
// process cascade.
package pipeline_test

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mheers/rethymno-emergency-pharmacy/internal/ocr"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/parse"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/pipeline"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/validate"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/vision"
)

var (
	workerBin  string
	remoteOnce sync.Once
	sharedR    *ocr.Remote
	sharedErr  error
)

// TestMain builds the worker binary once for the whole package and closes
// the shared worker afterwards.
func TestMain(m *testing.M) {
	if len(os.Args) > 1 && os.Args[1] == "ocr-worker" {
		fmt.Fprintln(os.Stderr, "pipeline test: refusing to run as OCR worker; tests must spawn the built rethymno-emergency-pharmacy binary, not this test binary")
		os.Exit(1)
	}
	dir, err := os.MkdirTemp("", "rethymno-emergency-pharmacy-worker-*")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	workerBin = filepath.Join(dir, "rethymno-emergency-pharmacy")
	build := exec.Command("go", "build", "-buildvcs=false", "-o", workerBin, "github.com/mheers/rethymno-emergency-pharmacy/cmd/rethymno-emergency-pharmacy")
	build.Stderr = os.Stderr
	if err := build.Run(); err != nil {
		os.RemoveAll(dir)
		fmt.Fprintln(os.Stderr, "build worker binary:", err)
		os.Exit(1)
	}
	code := m.Run()
	if sharedR != nil {
		sharedR.Close()
	}
	os.RemoveAll(dir)
	os.Exit(code)
}

// sharedWorker starts (once) the OCR worker subprocess used by all tests
// in this package.
func sharedWorker(t *testing.T, modelsDir string) *ocr.Remote {
	t.Helper()
	remoteOnce.Do(func() {
		sharedR, sharedErr = ocr.StartRemote([]string{workerBin, "ocr-worker"}, modelsDir, "", 0)
	})
	if sharedErr != nil {
		t.Skipf("OCR worker unavailable (run inside the Docker image): %v", sharedErr)
	}
	return sharedR
}

func loadSchedule(t *testing.T, path, modelsDir string) *pipeline.Result {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("image not present: %v", err)
	}
	v := vision.New(vision.Config{})
	remote := sharedWorker(t, modelsDir)
	pipe := pipeline.New(v, remote, nil)
	res, err := pipe.RunBytes(t.Context(), data, pipeline.Options{
		SourceURL: "https://fskriti.gr/efimeries-farmakeion-rethymnou/",
		City:      "Ρέθυμνο",
	})
	if err != nil {
		t.Fatal(err)
	}
	refs, err := validate.LoadReference(validate.ReferenceJSON)
	if err != nil {
		t.Fatal(err)
	}
	pipeline.FillFromCatalog(&res.Schedule, refs)
	vals, warnings := pipeline.ValidateResult(&res.Schedule, refs)
	res.Validations = vals
	res.Warnings = warnings
	pipeline.ApplyValidation(&res.Schedule, vals)
	return res
}

// repoRoot finds the repository root from the test working directory.
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 8; i++ {
		if _, err := os.Stat(dir + "/go.mod"); err == nil {
			return dir
		}
		idx := strings.LastIndex(dir, "/")
		if idx < 0 {
			break
		}
		dir = dir[:idx]
	}
	t.Fatal("repo root not found")
	return ""
}

// modelsDir finds the ONNX model directory from the repo root.
func modelsDir(t *testing.T) string {
	t.Helper()
	return repoRoot(t) + "/models"
}

// TestGoldenRealSchedule runs the pipeline on the real current-week image
// and verifies the expected structure: 8 days, day names matching the
// dates, and catalog-matching phone numbers.
func TestGoldenRealSchedule(t *testing.T) {
	modelsDir := modelsDir(t)
	if _, err := os.Stat(modelsDir); err != nil {
		t.Skip("models not present")
	}
	res := loadSchedule(t, repoRoot(t)+"/testdata/schedules/10.08.2026-17.08.2026_page-0001-2.jpg", modelsDir)

	sched := res.Schedule
	if len(sched.Days) != 8 {
		t.Fatalf("expected 8 days, got %d: %+v", len(sched.Days), dayList(sched.Days))
	}
	// day names must match the dates
	expected := map[string]string{
		"10/08/2026": "ΔΕΥΤΕΡΑ",
		"11/08/2026": "ΤΡΙΤΗ",
		"12/08/2026": "ΤΕΤΑΡΤΗ",
		"13/08/2026": "ΠΕΜΠΤΗ",
		"14/08/2026": "ΠΑΡΑΣΚΕΥΗ",
		"15/08/2026": "ΣΑΒΒΑΤΟ",
		"16/08/2026": "ΚΥΡΙΑΚΗ",
		"17/08/2026": "ΔΕΥΤΕΡΑ",
	}
	for _, d := range sched.Days {
		if want := expected[d.Date.Format("02/01/2006")]; want != "" && d.Day != want {
			t.Errorf("day %s: got %s want %s", d.Date.Format("02/01/2006"), d.Day, want)
		}
	}

	// every shift must have valid times and at least one pharmacy
	for _, d := range sched.Days {
		if len(d.Shifts) == 0 {
			t.Errorf("day %s has no shifts", d.Date.Format("02/01/2006"))
		}
		for _, s := range d.Shifts {
			if !validate.ValidateTime(s.From) || !validate.ValidateTime(s.To) {
				t.Errorf("day %s shift %s-%s invalid", d.Date.Format("02/01/2006"), s.From, s.To)
			}
			for _, p := range s.Pharmacies {
				if p.Phone == "" {
					t.Errorf("day %s pharmacy %s has no phone", d.Date.Format("02/01/2006"), p.Name)
				}
			}
		}
	}

	// known anchor: Monday 10/08 morning = Παπατζανή Μαρία 2831023347
	found := false
	for _, d := range sched.Days {
		if d.Date.Format("02/01/2006") != "10/08/2026" {
			continue
		}
		for _, s := range d.Shifts {
			for _, p := range s.Pharmacies {
				if p.Phone == "2831023347" {
					found = true
					if !strings.Contains(strings.ToUpper(p.Name), "ΠΑΠΑΤΖΑΝΗ") {
						t.Errorf("10/08 pharmacy name wrong: %q", p.Name)
					}
				}
			}
		}
	}
	if !found {
		t.Error("expected 2831023347 (Παπατζανή Μαρία) on Monday 10/08")
	}

	// deterministic JSON: two runs must produce identical bytes; wall-clock
	// stage timings are inherently nondeterministic and excluded
	res.Timings = pipeline.StageTimings{}
	out1, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	res2 := loadSchedule(t, repoRoot(t)+"/testdata/schedules/10.08.2026-17.08.2026_page-0001-2.jpg", modelsDir)
	res2.Timings = pipeline.StageTimings{}
	out2, err := json.MarshalIndent(res2, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if string(out1) != string(out2) {
		t.Error("JSON output is not deterministic across runs")
	}
}

// TestGoldenSecondWeek parses the next week's image as well.
func TestGoldenSecondWeek(t *testing.T) {
	modelsDir := modelsDir(t)
	if _, err := os.Stat(modelsDir); err != nil {
		t.Skip("models not present")
	}
	res := loadSchedule(t, repoRoot(t)+"/testdata/schedules/17.08.2026-24.08.2026_page-0001.jpg", modelsDir)
	if len(res.Schedule.Days) < 6 {
		t.Errorf("expected 6+ days for week 2, got %d", len(res.Schedule.Days))
	}
}

// TestGoldenThirdWeek covers the fixture containing duplicate catalog phone
// numbers and verifies that name-based disambiguation remains deterministic.
func TestGoldenThirdWeek(t *testing.T) {
	modelsDir := modelsDir(t)
	if _, err := os.Stat(modelsDir); err != nil {
		t.Skip("models not present")
	}
	res := loadSchedule(t, repoRoot(t)+"/testdata/schedules/24.08.2026-31.08.2026_page-0001.jpg", modelsDir)
	if len(res.Schedule.Days) != 8 {
		t.Fatalf("expected 8 days for week 3, got %d", len(res.Schedule.Days))
	}
	if len(res.Warnings) != 0 {
		t.Fatalf("expected catalog-clean week 3, got warnings: %v", res.Warnings)
	}
}

// TestDegradedOCRCaught feeds deliberately degraded OCR lines through the
// validator and asserts the validation catches the problems.
func TestDegradedOCRCaught(t *testing.T) {
	refs, _ := validate.LoadReference(validate.ReferenceJSON)
	v := validate.NewValidator(refs)

	// degraded phone: letter confusions and a missing digit
	res := v.ValidatePharmacy("ΠΑΠΑΤΖΑΝΗ ΜΑΡΙΑ", "ΓΕΡΑΚΑΡΗ 96", "2831O23347")
	if res.PhoneValid {
		t.Error("degraded phone must fail validation")
	}
	if len(res.Warnings) == 0 {
		t.Error("expected warnings for degraded phone")
	}

	// degraded name: no Greek letters
	res = v.ValidatePharmacy("XXXXX", "ΓΕΡΑΚΑΡΗ 96", "2831023347")
	if res.NameGreek {
		t.Error("non-Greek name must fail validation")
	}
	if res.CatalogMatch == nil {
		t.Error("phone still identifies the catalog entry")
	}

	// overnight shift must not be flagged invalid
	if !validate.ValidateTime("21:00") || !validate.ValidateTime("08:00") {
		t.Error("overnight shift times rejected")
	}
}

// TestFillFromCatalogPhoneNormalization verifies that catalog phones with
// irregular formatting (spaces) or multiple numbers still match OCR phones
// (plain digits).
func TestFillFromCatalogPhoneNormalization(t *testing.T) {
	refs := []validate.Reference{
		{Name: "Καλογεράκης Ιωάννης", Address: "Μοάτσου 8", Phone: "28310 22187"},
		{Name: "Μαστοράκη - Κεραμιανάκη", Address: "Λ. Πορτάλιου 20", Phone: "28310 51113, 28310 55649"},
	}
	sched := &parse.Schedule{Days: []parse.DaySchedule{{
		Date: time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC), Day: "ΔΕΥΤΕΡΑ",
		Shifts: []parse.Shift{{From: "08:00", To: "21:00", Pharmacies: []parse.Pharmacy{
			{Name: "", Address: "APKAΔIOY31", Phone: "2831022187"},
			{Name: "", Address: "", Phone: "2831099999"},
			{Name: "", Address: "MARKOYTOPTAAIOY20", Phone: "2831051113"},
		}}},
	}}}
	if n := pipeline.FillFromCatalog(sched, refs); n != 2 {
		t.Fatalf("expected 2 pharmacies filled from catalog, got %d", n)
	}
	if got := sched.Days[0].Shifts[0].Pharmacies[0].Name; got != "ΚΑΛΟΓΕΡΑΚΗΣ ΙΩΑΝΝΗΣ" {
		t.Errorf("catalog name fill wrong: got %q", got)
	}
	if got := sched.Days[0].Shifts[0].Pharmacies[2].Name; got != "ΜΑΣΤΟΡΑΚΗ - ΚΕΡΑΜΙΑΝΑΚΗ" {
		t.Errorf("dual-phone catalog fill wrong: got %q", got)
	}
	if got := sched.Days[0].Shifts[0].Pharmacies[1].Name; got != "" {
		t.Errorf("unmatched phone must not be filled, got %q", got)
	}
}

// TestFillFromCatalogEnrichesGeo verifies that matched pharmacies receive
// the Latin name and OpenStreetMap coordinates from the catalog.
func TestFillFromCatalogEnrichesGeo(t *testing.T) {
	refs := []validate.Reference{
		{Name: "Καλογεράκης Ιωάννης", Address: "Μοάτσου 8", Phone: "2831022187",
			NameLat: "Kalogerakis Ioannis", AddressLat: "Moatsou 8", Lat: 35.3641, Lon: 24.4736},
	}
	sched := &parse.Schedule{Days: []parse.DaySchedule{{
		Date: time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC), Day: "ΔΕΥΤΕΡΑ",
		Shifts: []parse.Shift{{From: "08:00", To: "21:00", Pharmacies: []parse.Pharmacy{
			{Name: "ΚΑΛΟΓΕΡΑΚΗΣ ΙΩΑΝΝΗΣ", Address: "ΜΟΑΤΣΟΥ 8", Phone: "28310 22187"},
			{Name: "Άγνωστο Φαρμακείο", Address: "Κάπου 1", Phone: "2831099999"},
		}}},
	}}}
	if n := pipeline.FillFromCatalog(sched, refs); n != 0 {
		t.Fatalf("names already present, expected 0 fills, got %d", n)
	}
	ph := sched.Days[0].Shifts[0].Pharmacies[0]
	if ph.NameLat != "Kalogerakis Ioannis" || ph.AddressLat != "Moatsou 8" || ph.Lat != 35.3641 || ph.Lon != 24.4736 {
		t.Errorf("catalog geo enrichment wrong: got %+v", ph)
	}
	unmatched := sched.Days[0].Shifts[0].Pharmacies[1]
	if unmatched.Lat != 0 || unmatched.Lon != 0 || unmatched.NameLat != "" || unmatched.AddressLat != "" {
		t.Errorf("unmatched pharmacy must not be enriched, got %+v", unmatched)
	}

	// the canonical date and the Latin fields must appear in the JSON output
	out, err := json.MarshalIndent(sched, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), `"date": "2026-08-17T00:00:00Z"`) {
		t.Errorf("date not canonical RFC 3339 in JSON: %s", out)
	}
	if !strings.Contains(string(out), `"name_latin": "Kalogerakis Ioannis"`) ||
		!strings.Contains(string(out), `"address_latin": "Moatsou 8"`) {
		t.Errorf("latin fields missing from JSON: %s", out)
	}
}

// TestFillFromCatalogEnrichesGoogle verifies that matched pharmacies receive
// the Google enrichment block (corrected coordinates, contact details,
// opening hours, photos) from the catalog, while unmatched and
// non-enriched entries stay clean.
func TestFillFromCatalogEnrichesGoogle(t *testing.T) {
	refs := []validate.Reference{
		{
			Name: "Δαφνομήλη Γεωργία", Address: "Δημοκρατίας 6", Phone: "2831056850",
			Google: &validate.GoogleDetails{
				PlaceID:            "ChIJbyS4JQh1mxQRLx7JVplGmAQ",
				FormattedAddress:   "Dimokratias 6, Rethymno 741 32, Greece",
				PhoneInternational: "+30 2831 056850",
				Website:            "https://dafnomili.example",
				GoogleMapsURL:      "https://maps.google.com/?cid=1",
				OpeningHours: &validate.OpenHours{
					WeekdayDescriptions: []string{"Monday: 8:30 AM - 3:00 PM"},
					Periods: []validate.Period{{
						Open:  &validate.PeriodPoint{Day: 1, Hour: 8, Minute: 30},
						Close: &validate.PeriodPoint{Day: 1, Hour: 15, Minute: 0},
					}},
				},
				Photos: []validate.PhotoDetail{{ContentType: "image/jpeg", Base64: "aGVsbG8="}},
			},
		},
		{Name: "Καλογεράκης Ιωάννης", Address: "Μοάτσου 8", Phone: "2831022187",
			NameLat: "Kalogerakis Ioannis", AddressLat: "Moatsou 8", Lat: 35.3641, Lon: 24.4736},
	}
	sched := &parse.Schedule{Days: []parse.DaySchedule{{
		Date: time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC), Day: "ΔΕΥΤΕΡΑ",
		Shifts: []parse.Shift{{From: "08:00", To: "21:00", Pharmacies: []parse.Pharmacy{
			{Name: "ΔΑΦΝΟΜΗΛΗ ΓΕΩΡΓΙΑ", Address: "ΔΗΜΟΚΡΑΤΙΑΣ 6", Phone: "2831056850"},
			{Name: "ΚΑΛΟΓΕΡΑΚΗΣ ΙΩΑΝΝΗΣ", Address: "ΜΟΑΤΣΟΥ 8", Phone: "2831022187"},
			{Name: "Άγνωστο Φαρμακείο", Address: "Κάπου 1", Phone: "2831099999"},
		}}},
	}}}
	if n := pipeline.FillFromCatalog(sched, refs); n != 0 {
		t.Fatalf("names already present, expected 0 fills, got %d", n)
	}

	enriched := sched.Days[0].Shifts[0].Pharmacies[0]
	if enriched.Google == nil {
		t.Fatal("enriched pharmacy missing google block")
	}
	if enriched.Google.PlaceID != "ChIJbyS4JQh1mxQRLx7JVplGmAQ" ||
		enriched.Google.Website != "https://dafnomili.example" ||
		enriched.Google.FormattedAddress != "Dimokratias 6, Rethymno 741 32, Greece" {
		t.Errorf("google enrichment wrong: %+v", enriched.Google)
	}
	if enriched.Google.OpeningHours == nil || len(enriched.Google.OpeningHours.Periods) != 1 {
		t.Errorf("opening hours missing: %+v", enriched.Google)
	}
	if len(enriched.Google.Photos) != 1 || enriched.Google.Photos[0].Base64 != "aGVsbG8=" {
		t.Errorf("photos missing: %+v", enriched.Google)
	}

	// non-enriched but matched pharmacy keeps OSM data and no google block
	osm := sched.Days[0].Shifts[0].Pharmacies[1]
	if osm.Google != nil {
		t.Errorf("non-enriched pharmacy must not get a google block: %+v", osm.Google)
	}
	if osm.Lat != 35.3641 || osm.Lon != 24.4736 {
		t.Errorf("OSM coordinates lost: %+v", osm)
	}

	// unmatched pharmacy stays clean
	unmatched := sched.Days[0].Shifts[0].Pharmacies[2]
	if unmatched.Google != nil || unmatched.Lat != 0 || unmatched.Lon != 0 {
		t.Errorf("unmatched pharmacy must not be enriched: %+v", unmatched)
	}

	// the google block must appear in the JSON output
	out, err := json.MarshalIndent(sched, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), `"google": {`) {
		t.Errorf("google block missing from JSON: %s", out)
	}
	if !strings.Contains(string(out), `"place_id": "ChIJbyS4JQh1mxQRLx7JVplGmAQ"`) {
		t.Errorf("place_id missing from JSON: %s", out)
	}
}

func dayList(days []parse.DaySchedule) []string {
	out := make([]string, 0, len(days))
	for _, d := range days {
		out = append(out, d.Date.Format("02/01/2006"))
	}
	return out
}
