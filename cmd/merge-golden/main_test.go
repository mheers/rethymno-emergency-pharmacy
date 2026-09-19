package main

import "testing"

func TestStrictError(t *testing.T) {
	cases := []struct {
		strict    bool
		curated   int
		fallbacks int
		wantErr   bool
	}{
		{false, 0, 0, false},
		{false, 3, 2, false},
		{true, 0, 0, false},
		{true, 1, 0, true},
		{true, 0, 1, true},
		{true, 2, 3, true},
	}
	for _, c := range cases {
		err := strictError(c.strict, c.curated, c.fallbacks)
		if (err != nil) != c.wantErr {
			t.Errorf("strictError(%v, %d, %d) = %v, wantErr %v",
				c.strict, c.curated, c.fallbacks, err, c.wantErr)
		}
	}
}
