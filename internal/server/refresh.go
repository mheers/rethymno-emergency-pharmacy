package server

import (
	"context"
	"time"

	rethymnoemergency "github.com/mheers/rethymno-emergency-pharmacy"
)

// refreshLoop warms the cache on startup, then refreshes the current week's
// schedule once a day at local midnight. Clients keep getting served from
// the cache in between; OCR never runs more than once per day after the
// startup warm-up.
func (s *Server) refreshLoop(ctx context.Context) {
	s.refreshNow(ctx)
	wait := s.nextRefresh
	for {
		t := time.NewTimer(time.Until(wait))
		select {
		case <-ctx.Done():
			t.Stop()
			return
		case <-t.C:
			s.refreshNow(ctx)
			wait = s.nextRefresh
		}
	}
}

// refreshNow unconditionally fetches and caches the current week's
// schedule. The fetch happens even when the cached entry looks fresh: the
// schedule can change mid-week (corrections, substitutions), so the cache
// is never trusted to be current without a fresh fetch. On success the
// next refresh is scheduled for the following local midnight; on failure
// it is retried at the next local midnight as well, keeping OCR runs at
// most one per day while the last known result keeps being served.
func (s *Server) refreshNow(ctx context.Context) {
	key := weekKey(s.now())
	refCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 2*time.Minute)
	defer cancel()
	e, err := s.fetchAndStore(refCtx, key, time.Time{})
	if err != nil {
		s.log.Printf("schedule refresh failed: %v; next refresh at local midnight", err)
		s.nextRefresh = nextMidnight(s.now())
		return
	}
	if e != nil {
		s.log.Printf("schedule refresh complete: %s (image sha256 %s); next refresh at local midnight",
			weekLabel(e.result), e.etag)
	} else {
		s.log.Print("schedule refresh complete; next refresh at local midnight")
	}
	s.nextRefresh = nextMidnight(s.now())
}

// weekLabel renders the week a result covers, for log lines.
func weekLabel(res *rethymnoemergency.Result) string {
	if res == nil || len(res.Schedule.Days) == 0 {
		return "no days parsed"
	}
	from := res.Schedule.Days[0].Date
	to := from.AddDate(0, 0, 6)
	return from.Format("02/01/2006") + ".." + to.Format("02/01/2006")
}

// nextMidnight returns the next 00:00 in the local time zone.
func nextMidnight(now time.Time) time.Time {
	return time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, now.Location()).AddDate(0, 0, 1)
}
