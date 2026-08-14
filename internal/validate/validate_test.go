package validate

import (
	"testing"

	"github.com/mheers/rethymno-emergency-pharmacy/internal/normalize"
)

func TestValidatePhone(t *testing.T) {
	for _, ok := range []string{"2831026115", "2831027264", "2834061450"} {
		if !ValidatePhone(ok) {
			t.Errorf("expected %s valid", ok)
		}
	}
	for _, bad := range []string{"12345", "3831026115", "283102611", "28310261155", "abc"} {
		if ValidatePhone(bad) {
			t.Errorf("expected %s invalid", bad)
		}
	}
}

func TestValidateTime(t *testing.T) {
	if !ValidateTime("08:00") || !ValidateTime("21:00") || !ValidateTime("23:59") {
		t.Error("valid times rejected")
	}
	for _, bad := range []string{"24:00", "8:00", "08:60", "08-00", ""} {
		if ValidateTime(bad) {
			t.Errorf("expected %q invalid", bad)
		}
	}
}

func TestValidateNameGreek(t *testing.T) {
	if !ValidateNameGreek("ΠΑΠΑΤΖΑΝΗ ΜΑΡΙΑ") {
		t.Error("greek name rejected")
	}
	if ValidateNameGreek("") || ValidateNameGreek("ABC") || ValidateNameGreek("ΠΑΠΑ 123") {
		t.Error("bad names accepted")
	}
}

func TestValidateAddress(t *testing.T) {
	for _, ok := range []string{"Γερακάρη 96", "Λ. Κουντουριώτου 81", "Αρκαδίου 92"} {
		if !ValidateAddress(ok) {
			t.Errorf("expected %q valid", ok)
		}
	}
}

func TestValidatorCrossCheck(t *testing.T) {
	refs := []Reference{
		{Name: "Αλεφαντινού Μαρία", Address: "Ζαμπελίου 19", Phone: "2831026115"},
		{Name: "Δαφνομήλη Γεωργία", Address: "Δημοκρατίας 6", Phone: "2831056850"},
	}
	v := NewValidator(refs)

	// exact phone match
	res := v.ValidatePharmacy("ΑΛΕΦΑΝΤΙΝΟΥ ΜΑΡΙΑ", "ΖΑΜΠΕΛΙΟΥ 19", "2831026115")
	if !res.PhoneValid || res.CatalogMatch == nil || res.Discrepancy != "" {
		t.Errorf("exact match failed: %+v", res)
	}
	if res.Score < 0.9 {
		t.Errorf("score too low: %f", res.Score)
	}

	// phone not in catalog but name matches: match reported with a
	// discrepancy, phone still flagged as valid
	res = v.ValidatePharmacy("ΑΛΕΦΑΝΤΙΝΟΥ ΜΑΡΙΑ", "ΖΑΜΠΕΛΙΟΥ 19", "2831099999")
	if res.CatalogMatch == nil {
		t.Error("name should match the catalog")
	}
	if res.Discrepancy == "" {
		t.Error("expected phone discrepancy")
	}
	if !res.PhoneValid {
		t.Error("valid unknown phone rejected")
	}

	// degraded OCR: wrong phone digit — name still matches but with a
	// discrepancy warning
	res = v.ValidatePharmacy("ΑΛΕΦΑΝΤΙΝΟΥ", "ΖΑΜΠΕΛΙΟΥ 19", "2831026116")
	if res.CatalogMatch == nil {
		t.Error("name should match the catalog")
	}
	if res.Discrepancy == "" {
		t.Error("expected discrepancy warning")
	}

	// degraded OCR: garbled name but right phone — catalog fills it
	res = v.ValidatePharmacy("AAEΦANTINOY", "ZAMEAIOY19", "2831026115")
	if res.CatalogMatch == nil {
		t.Error("garbled name with right phone should match catalog")
	}
}

func TestValidatorPhoneFormatting(t *testing.T) {
	refs := []Reference{
		{Name: "Καλογεράκης Ιωάννης", Address: "Μοάτσου 8", Phone: "28310 22187"},
		{Name: "Σπαντιδάκης Ιωσήφ", Address: "Πλ. Τεσσάρων Μαρτύρων", Phone: "28310 23666"},
		{Name: "Μαστοράκη - Κεραμιανάκη", Address: "Λ. Πορτάλιου 20", Phone: "28310 51113, 28310 55649"},
	}
	v := NewValidator(refs)

	// catalog phone stored with spaces; OCR emits plain digits
	res := v.ValidatePharmacy("ΚΑΛΟΓΕΡΑΚΗΣ ΙΩΑΝΝΗΣ", "ΜΟΑΤΣΟΥ 8", "2831022187")
	if res.CatalogMatch == nil || res.CatalogMatch.Phone != "28310 22187" {
		t.Errorf("space-formatted catalog phone not matched: %+v", res.CatalogMatch)
	}

	// OCR digit confusion (O -> 0) still matches the catalog phone
	res = v.ValidatePharmacy("ΣΠΑΝΤΙΔΑΚΗΣ ΙΩΣΗΦ", "ΠΛ. ΤΕΣΣΑΡΩΝ ΜΑΡΤΥΡΩΝ", "2831O23666")
	if res.CatalogMatch == nil || res.CatalogMatch.Phone != "28310 23666" {
		t.Errorf("OCR-confused phone not matched: %+v", res.CatalogMatch)
	}

	// dual-phone entry: either listed number must match
	res = v.ValidatePharmacy("ΜΑΣΤΟΡΑΚΗ ΚΕΡΑΜΙΑΝΑΚΗ", "Λ. ΠΟΡΤΑΛΙΟΥ 20", "2831051113")
	if res.CatalogMatch == nil || res.CatalogMatch.Phone != "28310 51113, 28310 55649" {
		t.Errorf("first dual-phone not matched: %+v", res.CatalogMatch)
	}
	res = v.ValidatePharmacy("ΜΑΣΤΟΡΑΚΗ ΚΕΡΑΜΙΑΝΑΚΗ", "Λ. ΠΟΡΤΑΛΙΟΥ 20", "2831055649")
	if res.CatalogMatch == nil {
		t.Errorf("second dual-phone not matched: %+v", res.CatalogMatch)
	}
}

func TestValidatorChoosesDuplicatePhoneByName(t *testing.T) {
	refs := []Reference{
		{Name: "Δαμβακεράκης", Address: "Λεωφ. Εμμανουήλ Παχλά 33", Phone: "2831054706"},
		{Name: "Τσιομπίκας Γρηγόριος", Address: "Εμμ. Παχλά 33 Περιβόλια", Phone: "28310 54706"},
	}
	v := NewValidator(refs)
	res := v.ValidatePharmacy("ΔΑΜΒΑΚΕΡΑΚΗΣ", "ΕΜΜΑΝΟΥΗΛ ΠΑΧΛΑ 33", "2831054706")
	if res.CatalogMatch == nil || res.CatalogMatch.Name != "Δαμβακεράκης" {
		t.Fatalf("duplicate phone chose wrong reference: %+v", res.CatalogMatch)
	}
	if res.Discrepancy != "" {
		t.Fatalf("duplicate phone produced discrepancy: %s", res.Discrepancy)
	}
}

func TestReferenceLoad(t *testing.T) {
	refs, err := LoadReference(ReferenceJSON)
	if err != nil {
		t.Fatal(err)
	}
	if len(refs) < 70 {
		t.Errorf("expected 70+ reference pharmacies, got %d", len(refs))
	}
	// every reference has a plausible phone
	withPhone := 0
	for _, r := range refs {
		if len(normalize.DigitsOnly(r.Phone)) == 10 {
			withPhone++
		}
	}
	if withPhone < 60 {
		t.Errorf("expected 60+ refs with 10-digit phones, got %d", withPhone)
	}
}

func TestReferenceManganiotisLocationAndAddress(t *testing.T) {
	refs, err := LoadReference(ReferenceJSON)
	if err != nil {
		t.Fatal(err)
	}

	byName := make(map[string]Reference, len(refs))
	for _, ref := range refs {
		byName[ref.Name] = ref
	}

	manganiotis, ok := byName["Μαγγανιώτης Σταμάτης"]
	if !ok {
		t.Fatal("Μαγγανιώτης Σταμάτης missing from reference catalog")
	}
	mastoraki, ok := byName["Μαστοράκη - Κεραμιανάκη"]
	if !ok {
		t.Fatal("Μαστοράκη - Κεραμιανάκη missing from reference catalog")
	}

	if manganiotis.Lat != 35.365122 || manganiotis.Lon != 24.48755 {
		t.Errorf("wrong Μαγγανιώτης coordinates: got %.6f, %.6f", manganiotis.Lat, manganiotis.Lon)
	}
	if manganiotis.Lat == mastoraki.Lat && manganiotis.Lon == mastoraki.Lon {
		t.Error("Μαγγανιώτης coordinates must not reuse Μαστοράκη coordinates")
	}
	if got, want := manganiotis.Address, "Αγγ. Σικελιανού 7 Πλατεία Αγ. Γεωργίου"; got != want {
		t.Errorf("wrong Μαγγανιώτης address: got %q, want %q", got, want)
	}

	if got, want := byName["Δρανδάκη - Λιάσκος"].Address, "Αρκαδίου 92, Ρέθυμνο"; got != want {
		t.Errorf("wrong Δρανδάκη address: got %q, want %q", got, want)
	}
	if got, want := byName["Νικολουδάκης Νικόλαος - Ευαγγελία"].Address, "Πλατεία Αγνώστου Στρατιώτη 31, Ρέθυμνο"; got != want {
		t.Errorf("wrong Νικολουδάκης address: got %q, want %q", got, want)
	}
}
