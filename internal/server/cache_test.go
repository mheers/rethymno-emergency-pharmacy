package server

import (
	"fmt"
	"testing"
	"time"
)

func TestCacheTTL(t *testing.T) {
	now := time.Now()
	c := newCache(func() time.Time { return now })
	e := &entry{etag: "abc", fetchedAt: now}
	c.put("week:2026-W33", e)

	if got := c.get("week:2026-W33"); got != e {
		t.Fatal("expected entry to be returned")
	}
	if !c.fresh(e, time.Hour) {
		t.Fatal("expected entry to be fresh")
	}
	now = now.Add(2 * time.Hour)
	if c.fresh(e, time.Hour) {
		t.Fatal("expected entry to be stale after TTL")
	}
}

func TestCacheSHAIndexBounded(t *testing.T) {
	c := newCache(time.Now)
	c.maxSHA = 3
	for i := 0; i < 5; i++ {
		c.put(fmt.Sprintf("week:key-%d", i), &entry{
			etag:      fmt.Sprintf("sha-%d", i),
			fetchedAt: time.Now(),
		})
	}
	if got := c.getSHA("sha-0"); got != nil {
		t.Fatal("sha-0 should have been evicted (bounded index)")
	}
	if got := c.getSHA("sha-4"); got == nil {
		t.Fatal("sha-4 should still be present")
	}
	if got := c.getSHA("unknown"); got != nil {
		t.Fatal("unknown sha should not be present")
	}
}

func TestWeekKey(t *testing.T) {
	monday := time.Date(2026, 8, 10, 12, 0, 0, 0, time.UTC)
	sunday := time.Date(2026, 8, 16, 12, 0, 0, 0, time.UTC)
	if weekKey(monday) != weekKey(sunday) {
		t.Fatalf("same ISO week must share a key: %s vs %s", weekKey(monday), weekKey(sunday))
	}
	nextWeek := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	if weekKey(monday) == weekKey(nextWeek) {
		t.Fatal("different ISO weeks must not share a key")
	}
}

func TestNextMidnight(t *testing.T) {
	now := time.Date(2026, 8, 14, 23, 59, 59, 0, time.Local)
	midnight := nextMidnight(now)
	want := time.Date(2026, 8, 15, 0, 0, 0, 0, time.Local)
	if !midnight.Equal(want) {
		t.Fatalf("nextMidnight(%v) = %v, want %v", now, midnight, want)
	}
}

func TestParseRequestDate(t *testing.T) {
	dd, err := parseRequestDate("15/08/2026")
	if err != nil || dd.Day() != 15 || dd.Month() != 8 || dd.Year() != 2026 {
		t.Fatalf("parseRequestDate(15/08/2026) = %v, %v", dd, err)
	}
	iso, err := parseRequestDate("2026-08-15")
	if err != nil || iso.Day() != 15 {
		t.Fatalf("parseRequestDate(2026-08-15) = %v, %v", iso, err)
	}
	if _, err := parseRequestDate("15.08.2026"); err == nil {
		t.Fatal("expected error for non-supported date format")
	}
}
