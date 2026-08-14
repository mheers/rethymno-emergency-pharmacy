package extract

import (
	"os"
	"strings"
	"testing"
	"time"
)

func TestFindScheduleImages(t *testing.T) {
	html := []byte(`
		<img src="https://fskriti.gr/wp-content/uploads/2022/05/10.08.2026-17.08.2026_page-0001-2.jpg">
		<img src="https://fskriti.gr/wp-content/uploads/2022/05/10.08.2026-17.08.2026_page-0001-2-1024x724.jpg">
		<img src="https://fskriti.gr/wp-content/uploads/2022/05/17.08.2026-24.08.2026_page-0001.jpg">
	`)
	imgs := FindScheduleImages(html)
	if len(imgs) != 2 {
		t.Fatalf("expected 2 full-size images, got %d", len(imgs))
	}
	if imgs[0].URL != "https://fskriti.gr/wp-content/uploads/2022/05/10.08.2026-17.08.2026_page-0001-2.jpg" {
		t.Errorf("first image wrong: %s", imgs[0].URL)
	}
	if !imgs[0].From.Equal(time.Date(2026, 8, 10, 0, 0, 0, 0, time.Local)) {
		t.Errorf("from date wrong: %v", imgs[0].From)
	}
	if !imgs[0].To.Equal(time.Date(2026, 8, 17, 0, 0, 0, 0, time.Local)) {
		t.Errorf("to date wrong: %v", imgs[0].To)
	}
}

func TestSelectCurrent(t *testing.T) {
	imgs := []ScheduleImage{
		{URL: "a", From: time.Date(2026, 8, 3, 0, 0, 0, 0, time.Local), To: time.Date(2026, 8, 10, 0, 0, 0, 0, time.Local)},
		{URL: "b", From: time.Date(2026, 8, 10, 0, 0, 0, 0, time.Local), To: time.Date(2026, 8, 17, 0, 0, 0, 0, time.Local)},
	}
	got, err := SelectCurrent(imgs, time.Date(2026, 8, 13, 12, 0, 0, 0, time.Local))
	if err != nil {
		t.Fatal(err)
	}
	if got.URL != "b" {
		t.Errorf("expected image b, got %s", got.URL)
	}
	// past the last image: falls back to newest
	got, err = SelectCurrent(imgs, time.Date(2026, 9, 1, 0, 0, 0, 0, time.Local))
	if err != nil {
		t.Fatal(err)
	}
	if got.URL != "b" {
		t.Errorf("expected fallback b, got %s", got.URL)
	}
}

func TestParseDutyPage(t *testing.T) {
	html := []byte(`
		<h3 class="duty-date">Τετάρτη 12 Αυγούστου 2026</h3>
		<table class="duties">
			<tr class="duty-time"><td colspan="2">08:00 πρωί - 21:00 βράδυ</td></tr>
			<tr><td class="pharmacy-title">Πρινιωτάκης Ηλίας - Παναγιώτης</td>
				<td class="pharmacy-address">Λ. Κουντουριώτου 81, <br />28310 27264</td></tr>
		</table>
		<table class="duties">
			<tr class="duty-time"><td colspan="2">21:00 βράδυ - 08:00 πρωί</td></tr>
			<tr><td class="pharmacy-title">Αλεφαντινού Μαρία</td>
				<td class="pharmacy-address">Ζαμπελίου 19<br />2831026115</td></tr>
		</table>
		<h3 class="duty-date">Πέμπτη 13 Αυγούστου 2026</h3>
	`)
	days := ParseDutyPage(html)
	if len(days) != 2 {
		t.Fatalf("expected 2 days, got %d", len(days))
	}
	d := days[0]
	if d.Date != "12/08/2026" || d.Day != "ΤΕΤΑΡΤΗ" {
		t.Errorf("day header wrong: %+v", d)
	}
	if len(d.Shifts) != 2 {
		t.Fatalf("expected 2 shifts, got %d", len(d.Shifts))
	}
	s0 := d.Shifts[0]
	if s0.From != "08:00" || s0.To != "21:00" {
		t.Errorf("shift times wrong: %+v", s0)
	}
	if len(s0.Pharmacies) != 1 || s0.Pharmacies[0].Phone != "2831027264" {
		t.Errorf("pharmacy wrong: %+v", s0.Pharmacies)
	}
	if s0.Pharmacies[0].Name != "Πρινιωτάκης Ηλίας - Παναγιώτης" {
		t.Errorf("name wrong: %q", s0.Pharmacies[0].Name)
	}
}

func TestParseCatalog(t *testing.T) {
	html := []byte(`
		<h3>Αγγελάκης Παναγιώτης | Ζωνιανά</h3>
		<p>Τηλέφωνο:</p><p>2834061450</p>
		<h3>Αλεφαντινού Μαρία | Ζαμπελίου 19</h3>
		<p>Εναντι κλ. Ασκληπιός</p>
		<p>Τηλέφωνο:</p><p>2831026115</p>
	`)
	refs := ParseCatalog(html)
	if len(refs) != 2 {
		t.Fatalf("expected 2 refs, got %d", len(refs))
	}
	if refs[0].Name != "Αγγελάκης Παναγιώτης" || refs[0].Phone != "2834061450" {
		t.Errorf("ref 0 wrong: %+v", refs[0])
	}
	if refs[1].Name != "Αλεφαντινού Μαρία" || refs[1].Phone != "2831026115" {
		t.Errorf("ref 1 wrong: %+v", refs[1])
	}
	if !strings.Contains(refs[1].Address, "Ζαμπελίου 19") {
		t.Errorf("ref 1 address wrong: %q", refs[1].Address)
	}
}

func TestParseRealDutyPage(t *testing.T) {
	data, err := os.ReadFile("../../testdata/reference/rethymno_pharmacies.json")
	if err != nil {
		t.Skip("reference file missing")
	}
	_ = data
	// the real HTML snapshot lives in testdata when present
	html, err := os.ReadFile("../../testdata/reference/rethymno_duty.html")
	if err != nil {
		t.Skip("no duty page snapshot")
	}
	days := ParseDutyPage(html)
	if len(days) == 0 {
		t.Fatal("no days parsed from real page")
	}
}
