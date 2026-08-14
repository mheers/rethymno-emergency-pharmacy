package normalize

import "testing"

func TestNormalizeGreek(t *testing.T) {
	cases := map[string]string{
		"ΔΕΥΤΕΡΑ":                "ΔΕΥΤΕΡΑ",
		"Παπατζανή Μαρία":         "Παπατζανη Μαρια",
		"08:00 - 21:00":          "08:00 - 21:00",
		"Λ. Κουντουριώτου 81,":   "Λ. Κουντουριωτου 81,",
		"Τρανταλίδου\u00a028":    "Τρανταλιδου 28",
		"Γερακάρη\u201496":       "Γερακαρη-96",
		"Γερακάρη – 96":          "Γερακαρη - 96",
		"\u201cΕΓΚΕΦΑΛΟΣ\u201d":  "\"ΕΓΚΕΦΑΛΟΣ\"",
		"ΑΛΕΦΑΝΤΙΝΟΥ\u200bΜΑΡΙΑ": "ΑΛΕΦΑΝΤΙΝΟΥ ΜΑΡΙΑ",
		"":                       "",
		"  spaces  here  ":       "spaces here",
	}
	for in, want := range cases {
		if got := NormalizeGreek(in); got != want {
			t.Errorf("NormalizeGreek(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestToGreekUppercase(t *testing.T) {
	cases := map[string]string{
		"ΔEYTEPA":      "ΔΕΥΤΕΡΑ",
		"THL":          "ΤΗΛ",
		"GERAKARI 96":  "ΓΕΡΑΚΑΡΙ 96",
		"KOUNTOURIOTI": "ΚΟΥΝΤΟΥΡΙΟΤΙ",
	}
	for in, want := range cases {
		if got := ToGreekUppercase(in); got != want {
			t.Errorf("ToGreekUppercase(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestDigitsOnly(t *testing.T) {
	cases := map[string]string{
		"THA.2831023347": "2831023347",
		"28310 34458":    "2831034458",
		"THA2831025123":  "2831025123",
		"":               "",
		"abc":            "",
		"Ο28310Ο27264":    "028310027264",
		"I2831O34458I":    "128310344581",
		"B2B31027264":     "82831027264",
	}
	for in, want := range cases {
		if got := DigitsOnly(in); got != want {
			t.Errorf("DigitsOnly(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestGreekToLatin(t *testing.T) {
	cases := map[string]string{
		"ΔΑΦΝΟΜΗΛΗ":    "DAFNOMILI",
		"ΚΟΥΝΤΟΥΡΙΩΤΟΥ": "KOUNTOURIOTOU",
		"ΑΛΕΦΑΝΤΙΝΟΥ":  "ALEFANTINOU",
		"ΠΑΠΑΤΖΑΝΗ":    "PAPATZANI",
	}
	for in, want := range cases {
		if got := GreekToLatin(in); got != want {
			t.Errorf("GreekToLatin(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestSimilarity(t *testing.T) {
	if s := Similarity("ALEFANTINOU", "ALEFANTINOU"); s != 1.0 {
		t.Errorf("identical strings: got %f", s)
	}
	if s := Similarity("ALEFANTINOU", "ALEFANTINOY"); s < 0.8 {
		t.Errorf("close strings: got %f", s)
	}
	if s := Similarity("ABC", "XYZ"); s > 0.1 {
		t.Errorf("far strings: got %f", s)
	}
}
