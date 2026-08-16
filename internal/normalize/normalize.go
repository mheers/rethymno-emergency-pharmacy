// Package normalize implements Greek text normalization and
// domain-specific OCR correction for the Rethymno pharmacy schedules.
//
// The PP-OCRv6 recognition model frequently outputs a mixture of Greek,
// Latin look-alike letters, and transliterated Greek (e.g. "GERAKARI" for
// "ΓΕΡΑΚΑΡΗ"). This package canonicalizes such output without destroying
// the original OCR result.
package normalize

import (
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// GreekLookAlike maps Latin letters that appear inside Greek text to their
// visually-equivalent Greek capital letters. PP-OCRv6 mixes these liberally
// ("ΔEYTEPA", "THA" for "ΤΗΛ").
var GreekLookAlike = map[rune]rune{
	'A': 'Α', 'B': 'Β', 'C': 'Κ', 'D': 'Δ', 'E': 'Ε', 'F': 'Φ', 'G': 'Γ',
	'H': 'Η', 'I': 'Ι', 'J': 'Ξ', 'K': 'Κ', 'L': 'Λ', 'M': 'Μ', 'N': 'Ν',
	'O': 'Ο', 'P': 'Ρ', 'Q': 'Κ', 'R': 'Ρ', 'S': 'Σ', 'T': 'Τ', 'U': 'Υ',
	'V': 'Β', 'W': 'Ω', 'X': 'Χ', 'Y': 'Υ', 'Z': 'Ζ',
}

// DigitConfusions maps letters that OCR commonly confuses with digits to
// their numeric value. Only meaningful in digit contexts (phone numbers).
var DigitConfusions = map[rune]rune{
	'O': '0', 'o': '0', 'Ο': '0', 'ο': '0', 'Θ': '0', 'θ': '0', 'Q': '0',
	'I': '1', 'i': '1', 'Ι': '1', 'ι': '1', 'l': '1', 'L': '1', '|': '1',
	'Z': '2', 'z': '2', 'S': '5', 's': '5', 'G': '6', 'g': '6', 'B': '8',
	'b': '8', 'β': '8',
}

var (
	dashVariants   = map[rune]bool{'‐': true, '‑': true, '‒': true, '–': true, '—': true, '―': true, '−': true, '⁻': true, '﹣': true, '－': true, '_': true}
	quoteVariants  = map[rune]rune{'\u2018': '\'', '\u2019': '\'', '\u201a': '\'', '\u201b': '\'', '\u201c': '"', '\u201d': '"', '\u201e': '"', '\u201f': '"', '«': '"', '»': '"'}
	spaceVariants  = map[rune]bool{'\u00a0': true, '\u2009': true, '\u200a': true, '\u202f': true, '\u2003': true, '\u2002': true, '\u200b': true, '\ufeff': true}
	greekToneChars = map[rune]rune{
		'ά': 'α', 'έ': 'ε', 'ή': 'η', 'ί': 'ι', 'ϊ': 'ι', 'ΐ': 'ι', 'ό': 'ο',
		'ύ': 'υ', 'ϋ': 'υ', 'ΰ': 'υ', 'ώ': 'ω', 'ᾶ': 'α', 'ῆ': 'η', 'ῖ': 'ι',
		'ῦ': 'υ', 'ῶ': 'ω', 'ἀ': 'α', 'ἁ': 'α', 'ἂ': 'α', 'ἃ': 'α', 'ἄ': 'α',
		'ἅ': 'α', 'ἆ': 'α', 'ἇ': 'α', 'ἐ': 'ε', 'ἑ': 'ε', 'ἒ': 'ε', 'ἓ': 'ε',
		'ἔ': 'ε', 'ἕ': 'ε', 'ἠ': 'η', 'ἡ': 'η', 'ἢ': 'η', 'ἣ': 'η', 'ἤ': 'η',
		'ἥ': 'η', 'ἦ': 'η', 'ἧ': 'η', 'ἰ': 'ι', 'ἱ': 'ι', 'ἲ': 'ι', 'ἳ': 'ι',
		'ἴ': 'ι', 'ἵ': 'ι', 'ἶ': 'ι', 'ἷ': 'ι', 'ὀ': 'ο', 'ὁ': 'ο', 'ὂ': 'ο',
		'ὃ': 'ο', 'ὄ': 'ο', 'ὅ': 'ο', 'ὐ': 'υ', 'ὑ': 'υ', 'ὒ': 'υ', 'ὓ': 'υ',
		'ὔ': 'υ', 'ὕ': 'υ', 'ὖ': 'υ', 'ὗ': 'υ', 'ὠ': 'ω', 'ὡ': 'ω', 'ὢ': 'ω',
		'ὣ': 'ω', 'ὤ': 'ω', 'ὥ': 'ω', 'ὦ': 'ω', 'ὧ': 'ω', 'ὰ': 'α', 'ὲ': 'ε',
		'ὴ': 'η', 'ὶ': 'ι', 'ὸ': 'ο', 'ὺ': 'υ', 'ὼ': 'ω', 'ᾳ': 'α', 'ῃ': 'η',
		'ῳ': 'ω', 'ᾀ': 'α', 'ᾁ': 'α', 'ᾂ': 'α', 'ᾃ': 'α', 'ᾄ': 'α', 'ᾅ': 'α',
		'ᾆ': 'α', 'ᾇ': 'α', 'ᾐ': 'η', 'ᾑ': 'η', 'ᾒ': 'η', 'ᾓ': 'η', 'ᾔ': 'η',
		'ᾕ': 'η', 'ᾖ': 'η', 'ᾗ': 'η', 'ᾠ': 'ω', 'ᾡ': 'ω', 'ᾢ': 'ω', 'ᾣ': 'ω',
		'ᾤ': 'ω', 'ᾥ': 'ω', 'ᾦ': 'ω', 'ᾧ': 'ω', 'Ά': 'Α', 'Έ': 'Ε', 'Ή': 'Η',
		'Ί': 'Ι', 'Ϊ': 'Ι', 'Ό': 'Ο', 'Ύ': 'Υ', 'Ϋ': 'Υ', 'Ώ': 'Ω',
	}
)

// NormalizeGreek performs NFC normalization, collapses whitespace,
// normalizes dash/quote variants, and strips Greek accents in a way that
// preserves NFC form. It does not change letters or digits.
func NormalizeGreek(s string) string {
	s = norm.NFC.String(s)
	var b strings.Builder
	b.Grow(len(s))
	lastSpace := false
	for _, r := range s {
		switch {
		case spaceVariants[r]:
			if !lastSpace {
				b.WriteByte(' ')
				lastSpace = true
			}
		case dashVariants[r]:
			b.WriteByte('-')
			lastSpace = false
		case r == ' ' || r == '\t' || r == '\n' || r == '\r':
			if !lastSpace {
				b.WriteByte(' ')
				lastSpace = true
			}
		default:
			if q, ok := quoteVariants[r]; ok {
				b.WriteRune(q)
			} else if g, ok := greekToneChars[r]; ok {
				b.WriteRune(g)
			} else {
				b.WriteRune(r)
			}
			lastSpace = false
		}
	}
	return strings.TrimSpace(b.String())
}

// ToGreekUppercase converts the string to Greek-styled uppercase, mapping
// Latin look-alike letters to their Greek counterparts. This is intended
// for pharmacy names and addresses where the OCR mixes scripts.
func ToGreekUppercase(s string) string {
	var b strings.Builder
	for _, r := range s {
		r = unicode.ToUpper(r)
		if g, ok := GreekLookAlike[r]; ok {
			b.WriteRune(g)
		} else {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// DigitsOnly extracts a phone-number-like digit string from s, mapping
// OCR letter confusions (O→0, B→8, I→1, ...) to digits and dropping
// everything else. Returns "" when fewer than 6 digits remain.
func DigitsOnly(s string) string {
	var b strings.Builder
	for _, r := range s {
		if r >= '0' && r <= '9' {
			b.WriteRune(r)
			continue
		}
		if d, ok := DigitConfusions[r]; ok {
			b.WriteRune(d)
		}
	}
	d := b.String()
	if len(d) < 6 {
		return ""
	}
	return d
}

// Phones splits a comma-separated phone field (as found in the reference
// catalog, e.g. "28310 51113, 28310 55649") into its individual
// digit-normalized numbers.
func Phones(s string) []string {
	out := make([]string, 0, 2)
	for _, p := range strings.Split(s, ",") {
		if d := DigitsOnly(p); d != "" {
			out = append(out, d)
		}
	}
	return out
}

// GreekToLatin transliterates Greek capital letters to canonical Greeklish
// used for fuzzy matching against the reference pharmacy catalog.
func GreekToLatin(s string) string {
	s = norm.NFC.String(s)
	s = strings.ToUpper(s)
	rs := []rune(s)
	// Strip Greek accents/tone marks so accented letters hit the base table.
	for i := range rs {
		if base, ok := greekToneChars[rs[i]]; ok {
			rs[i] = base
		}
	}
	var b strings.Builder
	for i := 0; i < len(rs); i++ {
		if i+1 < len(rs) {
			switch string(rs[i : i+2]) {
			case "ΑΥ":
				b.WriteString("AV")
				i++
				continue
			case "ΕΥ":
				b.WriteString("EV")
				i++
				continue
			case "ΟΥ":
				b.WriteString("OU")
				i++
				continue
			case "ΤΖ":
				b.WriteString("TZ")
				i++
				continue
			case "ΓΓ":
				b.WriteString("NG")
				i++
				continue
			case "ΓΚ":
				b.WriteString("GK")
				i++
				continue
			case "ΜΠ":
				b.WriteString("MP")
				i++
				continue
			}
		}
		switch rs[i] {
		case 'Α':
			b.WriteByte('A')
		case 'Β':
			b.WriteByte('V')
		case 'Γ':
			b.WriteByte('G')
		case 'Δ':
			b.WriteByte('D')
		case 'Ε':
			b.WriteByte('E')
		case 'Ζ':
			b.WriteByte('Z')
		case 'Η':
			b.WriteByte('I')
		case 'Θ':
			b.WriteString("TH")
		case 'Ι':
			b.WriteByte('I')
		case 'Κ':
			b.WriteByte('K')
		case 'Λ':
			b.WriteByte('L')
		case 'Μ':
			b.WriteByte('M')
		case 'Ν':
			b.WriteByte('N')
		case 'Ξ':
			b.WriteString("KS")
		case 'Ο':
			b.WriteByte('O')
		case 'Π':
			b.WriteByte('P')
		case 'Ρ':
			b.WriteByte('R')
		case 'Σ':
			b.WriteByte('S')
		case 'Τ':
			b.WriteByte('T')
		case 'Υ':
			b.WriteByte('Y')
		case 'Φ':
			b.WriteByte('F')
		case 'Χ':
			b.WriteString("CH")
		case 'Ψ':
			b.WriteString("PS")
		case 'Ω':
			b.WriteByte('O')
		default:
			b.WriteRune(rs[i])
		}
	}
	return b.String()
}

// Similarity returns a normalized [0,1] similarity between two strings
// using a length-aware edit distance. Used for fuzzy matching OCR output
// against the reference catalog.
func Similarity(a, b string) float64 {
	a = strings.ToUpper(strings.TrimSpace(a))
	b = strings.ToUpper(strings.TrimSpace(b))
	if a == b {
		return 1.0
	}
	if a == "" || b == "" {
		return 0.0
	}
	d := levenshtein(a, b)
	maxLen := len(a)
	if len(b) > maxLen {
		maxLen = len(b)
	}
	return 1.0 - float64(d)/float64(maxLen)
}

func levenshtein(a, b string) int {
	ra, rb := []rune(a), []rune(b)
	prev := make([]int, len(rb)+1)
	cur := make([]int, len(rb)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ra); i++ {
		cur[0] = i
		for j := 1; j <= len(rb); j++ {
			cost := 0
			if ra[i-1] != rb[j-1] {
				cost = 1
			}
			del := prev[j] + 1
			ins := cur[j-1] + 1
			sub := prev[j-1] + cost
			cur[j] = min3(del, ins, sub)
		}
		prev, cur = cur, prev
	}
	return prev[len(rb)]
}

func min3(a, b, c int) int {
	if a < b {
		if a < c {
			return a
		}
		return c
	}
	if b < c {
		return b
	}
	return c
}
