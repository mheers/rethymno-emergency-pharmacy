package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	rethymnoemergency "github.com/mheers/rethymno-emergency-pharmacy"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/parse"
)

// mondayOf returns the Monday of the week containing now.
func mondayOf(now time.Time) time.Time {
	for now.Weekday() != time.Monday {
		now = now.AddDate(0, 0, -1)
	}
	return time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, now.Location())
}

// testResult builds a result whose schedule covers the full current week
// (Mon..Sun), like a real parsed schedule.
func testResult(n int) *rethymnoemergency.Result {
	monday := mondayOf(time.Now())
	days := make([]parse.DaySchedule, 7)
	for i := range days {
		days[i] = parse.DaySchedule{
			Date: monday.AddDate(0, 0, i),
			Day:  "ΔΕΥΤΕΡΑ",
			Shifts: []parse.Shift{{
				From: "08:00", To: "21:00",
				Pharmacies: []parse.Pharmacy{{
					Name:       fmt.Sprintf("ΦΑΡΜΑΚΕΙΟ %d", n),
					Address:    "ΓΕΡΑΚΑΡΗ 96",
					Phone:      "2831023347",
					Confidence: 0.9,
				}},
			}},
		}
	}
	return &rethymnoemergency.Result{
		Schedule:    parse.Schedule{City: "Ρέθυμνο", Days: days},
		ImageSHA256: fmt.Sprintf("sha-%d", n),
	}
}

// newTestServer builds a Server with a stubbed fetch+parse unit and a
// controllable clock; the returned advance() moves the clock forward by one
// hour.
func newTestServer(fetchParse func(ctx context.Context, date time.Time) (*rethymnoemergency.Result, error)) (*Server, func(), func()) {
	now := time.Now()
	advance := func() { now = now.Add(time.Hour) }
	srv := New(Config{})
	srv.now = func() time.Time { return now }
	srv.cache.now = func() time.Time { return now }
	srv.fetchParse = func(ctx context.Context, date time.Time) (*rethymnoemergency.Result, error) {
		if fetchParse == nil {
			return testResult(1), nil
		}
		return fetchParse(ctx, date)
	}
	srv.parseImage = func(ctx context.Context, data []byte, opts rethymnoemergency.Options) (*rethymnoemergency.Result, error) {
		return testResult(1), nil
	}
	return srv, advance, func() {}
}

func get(t *testing.T, srv *Server, method, path string, body func(w *multipart.Writer)) *httptest.ResponseRecorder {
	t.Helper()
	var req *http.Request
	if body == nil {
		req = httptest.NewRequest(method, path, nil)
	} else {
		var buf bytes.Buffer
		w := multipart.NewWriter(&buf)
		body(w)
		w.Close()
		req = httptest.NewRequest(method, path, &buf)
		req.Header.Set("Content-Type", w.FormDataContentType())
	}
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, req) //nolint:errcheck
	return rr
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("GET /catalog", s.handleCatalog)
	mux.HandleFunc("GET /schedule/current", s.handleCurrent)
	mux.HandleFunc("GET /schedule/week", s.handleWeek)
	mux.HandleFunc("GET /schedule/image", s.handleImage)
	mux.HandleFunc("POST /parse", s.handleParse)
	mux.ServeHTTP(w, r)
}

func TestHealth(t *testing.T) {
	srv, _, _ := newTestServer(nil)
	rr := get(t, srv, "GET", "/healthz", nil)
	if rr.Code != http.StatusOK || !strings.Contains(rr.Body.String(), `"ok":true`) {
		t.Fatalf("healthz: code=%d body=%s", rr.Code, rr.Body.String())
	}
}

// TestCatalog serves the embedded golden catalog with lat/lon on every entry.
func TestCatalog(t *testing.T) {
	srv, _, _ := newTestServer(nil)
	rr := get(t, srv, "GET", "/catalog", nil)
	if rr.Code != http.StatusOK {
		t.Fatalf("catalog: code=%d body=%s", rr.Code, rr.Body.String())
	}
	var entries []struct {
		Name string  `json:"name"`
		Lat  float64 `json:"lat"`
		Lon  float64 `json:"lon"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &entries); err != nil {
		t.Fatalf("catalog JSON: %v", err)
	}
	if len(entries) < 10 {
		t.Fatalf("catalog has %d entries, want >= 10", len(entries))
	}
	for _, entry := range entries {
		if entry.Name == "" {
			t.Fatalf("catalog entry missing name: %+v", entry)
		}
		if entry.Lat == 0 || entry.Lon == 0 {
			t.Fatalf("catalog entry %q missing lat/lon: %+v", entry.Name, entry)
		}
	}
	etag := rr.Header().Get("ETag")
	if etag == "" {
		t.Fatal("catalog response missing ETag")
	}
	req := httptest.NewRequest("GET", "/catalog", nil)
	req.Header.Set("If-None-Match", etag)
	rr2 := httptest.NewRecorder()
	srv.ServeHTTP(rr2, req) //nolint:errcheck
	if rr2.Code != http.StatusNotModified {
		t.Fatalf("catalog If-None-Match: expected 304, got %d", rr2.Code)
	}
}

func TestCurrentFetchesOnceThenServesFromCache(t *testing.T) {
	var mu sync.Mutex
	calls := 0
	srv, _, _ := newTestServer(func(ctx context.Context, date time.Time) (*rethymnoemergency.Result, error) {
		mu.Lock()
		calls++
		mu.Unlock()
		return testResult(calls), nil
	})

	rr := get(t, srv, "GET", "/schedule/current", nil)
	if rr.Code != http.StatusOK {
		t.Fatalf("first request: code=%d body=%s", rr.Code, rr.Body.String())
	}
	if got := rr.Header().Get("X-Served-From"); got != "" {
		t.Fatalf("first request must be live, got X-Served-From=%q", got)
	}
	first := rr.Body.String()
	if !strings.Contains(first, "ΦΑΡΜΑΚΕΙΟ 1") {
		t.Fatalf("first request body: %s", first)
	}

	rr = get(t, srv, "GET", "/schedule/current", nil)
	if got := rr.Header().Get("X-Served-From"); got != "cache" {
		t.Fatalf("second request must be cached, got %q", got)
	}
	if rr.Body.String() != first {
		t.Fatal("cached body must be byte-identical")
	}
	mu.Lock()
	if calls != 1 {
		t.Fatalf("expected 1 fetch, got %d", calls)
	}
	mu.Unlock()
}

func TestIfNoneMatch(t *testing.T) {
	srv, _, _ := newTestServer(nil)
	first := get(t, srv, "GET", "/schedule/current", nil)
	etag := first.Header().Get("ETag")
	if etag != `"sha-1"` {
		t.Fatalf("ETag = %q", etag)
	}
	req := httptest.NewRequest("GET", "/schedule/current", nil)
	req.Header.Set("If-None-Match", etag)
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, req) //nolint:errcheck
	if rr.Code != http.StatusNotModified {
		t.Fatalf("expected 304, got %d", rr.Code)
	}
}

func TestWeekByDate(t *testing.T) {
	srv, _, _ := newTestServer(nil)
	// a date inside the current ISO week (Monday-based), so the week key
	// is shared with /schedule/current
	inWeek := mondayOf(time.Now()).AddDate(0, 0, 2)
	rr := get(t, srv, "GET", "/schedule/week?date="+inWeek.Format("02/01/2006"), nil)
	if rr.Code != http.StatusOK {
		t.Fatalf("week: code=%d body=%s", rr.Code, rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), "ΦΑΡΜΑΚΕΙΟ 1") {
		t.Fatalf("week body: %s", rr.Body.String())
	}
	// the same week shares a cache entry with /schedule/current
	rr = get(t, srv, "GET", "/schedule/current", nil)
	if rr.Header().Get("X-Served-From") != "cache" {
		t.Fatal("week and current requests for the same ISO week must share the cache")
	}

	rr = get(t, srv, "GET", "/schedule/week", nil)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("missing date: expected 400, got %d", rr.Code)
	}
	rr = get(t, srv, "GET", "/schedule/week?date=nonsense", nil)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("bad date: expected 400, got %d", rr.Code)
	}
}

func TestImageBySHA(t *testing.T) {
	srv, _, _ := newTestServer(nil)
	get(t, srv, "GET", "/schedule/current", nil)
	rr := get(t, srv, "GET", "/schedule/image?sha256=sha-1", nil)
	if rr.Code != http.StatusOK || !strings.Contains(rr.Body.String(), "ΦΑΡΜΑΚΕΙΟ 1") {
		t.Fatalf("image: code=%d body=%s", rr.Code, rr.Body.String())
	}
	rr = get(t, srv, "GET", "/schedule/image?sha256=unknown", nil)
	if rr.Code != http.StatusNotFound {
		t.Fatalf("unknown sha: expected 404, got %d", rr.Code)
	}
}

func TestParseMultipart(t *testing.T) {
	srv, _, _ := newTestServer(nil)
	rr := get(t, srv, "POST", "/parse", func(w *multipart.Writer) {
		fw, err := w.CreateFormFile("image", "schedule.jpg")
		if err != nil {
			t.Fatal(err)
		}
		fw.Write([]byte("fake-jpeg-bytes"))
	})
	if rr.Code != http.StatusOK {
		t.Fatalf("parse: code=%d body=%s", rr.Code, rr.Body.String())
	}
	if rr.Header().Get("X-Served-From") == "cache" {
		t.Fatal("on-demand parse must not come from the cache")
	}
	rr = get(t, srv, "POST", "/parse", nil)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("missing image: expected 400, got %d", rr.Code)
	}
}

// TestStaleIsServedWithoutRevalidation pins the OCR budget contract: OCR
// runs at most once per day, so a stale request is served from the cache
// and never triggers a background revalidation.
func TestStaleIsServedWithoutRevalidation(t *testing.T) {
	var mu sync.Mutex
	calls := 0
	srv, advance, _ := newTestServer(func(ctx context.Context, date time.Time) (*rethymnoemergency.Result, error) {
		mu.Lock()
		calls++
		n := calls
		mu.Unlock()
		return testResult(n), nil
	})
	srv.cfg.CacheTTL = time.Hour

	first := get(t, srv, "GET", "/schedule/current", nil)
	if !strings.Contains(first.Body.String(), "ΦΑΡΜΑΚΕΙΟ 1") {
		t.Fatalf("first body: %s", first.Body.String())
	}

	advance() // move the clock past the TTL
	rr := get(t, srv, "GET", "/schedule/current", nil)
	if rr.Code != http.StatusOK {
		t.Fatalf("stale request: code=%d", rr.Code)
	}
	if rr.Header().Get("X-Served-From") != "cache" {
		t.Fatal("stale request must still be served from cache")
	}
	// OCR must not run again: the daily refresh is the only writer.
	time.Sleep(50 * time.Millisecond)
	mu.Lock()
	defer mu.Unlock()
	if calls != 1 {
		t.Fatalf("stale request must not trigger a revalidation, got %d fetches", calls)
	}
}

func TestSingleFlight(t *testing.T) {
	release := make(chan struct{})
	started := make(chan struct{}, 1)
	var mu sync.Mutex
	calls := 0
	srv, _, _ := newTestServer(func(ctx context.Context, date time.Time) (*rethymnoemergency.Result, error) {
		mu.Lock()
		calls++
		mu.Unlock()
		select {
		case started <- struct{}{}:
		default:
		}
		<-release
		return testResult(calls), nil
	})

	var wg sync.WaitGroup
	codes := make([]int, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			req := httptest.NewRequest("GET", "/schedule/current", nil)
			rr := httptest.NewRecorder()
			srv.ServeHTTP(rr, req) //nolint:errcheck
			codes[i] = rr.Code
		}(i)
	}
	<-started
	close(release)
	wg.Wait()

	for i, c := range codes {
		if c != http.StatusOK {
			t.Fatalf("request %d: code=%d", i, c)
		}
	}
	mu.Lock()
	if calls != 1 {
		t.Fatalf("single flight violated: %d fetches", calls)
	}
	mu.Unlock()
}

func TestRefreshNowSchedules(t *testing.T) {
	now := time.Date(2026, 8, 14, 10, 0, 0, 0, time.Local)
	srv, _, _ := newTestServer(nil)
	srv.now = func() time.Time { return now }
	srv.nextRefresh = now

	srv.refreshNow(context.Background())
	if !srv.nextRefresh.Equal(nextMidnight(now)) {
		t.Fatalf("success: next refresh %v, want midnight %v", srv.nextRefresh, nextMidnight(now))
	}

	srv.fetchParse = func(ctx context.Context, date time.Time) (*rethymnoemergency.Result, error) {
		return nil, fmt.Errorf("upstream down")
	}
	srv.refreshNow(context.Background())
	if want := nextMidnight(now); !srv.nextRefresh.Equal(want) {
		t.Fatalf("failure: next refresh %v, want %v", srv.nextRefresh, want)
	}
	// the stale entry from the earlier success must still be served
	rr := get(t, srv, "GET", "/schedule/current", nil)
	if rr.Code != http.StatusOK || rr.Header().Get("X-Served-From") != "cache" {
		t.Fatalf("stale entry must survive refresh failures: code=%d served=%q", rr.Code, rr.Header().Get("X-Served-From"))
	}
}

// TestRefreshAlwaysRefetchesEvenWhenFresh pins the core freshness contract:
// the daily refresh must fetch the upstream unconditionally, even when the
// cached entry looks fresh, because the schedule can change mid-week.
func TestRefreshAlwaysRefetchesEvenWhenFresh(t *testing.T) {
	var mu sync.Mutex
	calls := 0
	srv, _, _ := newTestServer(func(ctx context.Context, date time.Time) (*rethymnoemergency.Result, error) {
		mu.Lock()
		calls++
		mu.Unlock()
		return testResult(calls), nil
	})

	get(t, srv, "GET", "/schedule/current", nil)
	mu.Lock()
	if calls != 1 {
		t.Fatalf("expected 1 fetch before refresh, got %d", calls)
	}
	mu.Unlock()

	srv.refreshNow(context.Background()) // entry is fresh here — must still refetch

	mu.Lock()
	if calls != 2 {
		t.Fatalf("daily refresh must refetch unconditionally, got %d fetches", calls)
	}
	mu.Unlock()
}

// TestFreshFallbackIsServedWithoutHammering: when the week's image is not
// published yet, the newest (previous-week) result is the best available
// answer and must be served without per-request refetches — the daily
// refresh replaces it once the new image appears.
func TestFreshFallbackIsServedWithoutHammering(t *testing.T) {
	var mu sync.Mutex
	calls := 0
	lastWeek := mondayOf(time.Now()).AddDate(0, 0, -7)
	srv, _, _ := newTestServer(func(ctx context.Context, date time.Time) (*rethymnoemergency.Result, error) {
		mu.Lock()
		calls++
		mu.Unlock()
		res := testResult(calls)
		res.Schedule.Days[0].Date = lastWeek // does NOT cover today
		return res, nil
	})

	get(t, srv, "GET", "/schedule/current", nil) // miss -> blocking fetch
	get(t, srv, "GET", "/schedule/current", nil) // fresh fallback -> served, no refetch
	get(t, srv, "GET", "/schedule/current", nil) // still fresh -> served, no refetch
	mu.Lock()
	if calls != 1 {
		t.Fatalf("fresh fallback must not trigger per-request refetches, got %d", calls)
	}
	mu.Unlock()
}

// TestStaleNonCoveringIsServedFromCache: once the fallback goes stale it is
// still the best available answer and must be served from the cache — OCR
// never runs from the request path, so no blocking refetch happens.
func TestStaleNonCoveringIsServedFromCache(t *testing.T) {
	var mu sync.Mutex
	calls := 0
	lastWeek := mondayOf(time.Now()).AddDate(0, 0, -7)
	srv, advance, _ := newTestServer(func(ctx context.Context, date time.Time) (*rethymnoemergency.Result, error) {
		mu.Lock()
		calls++
		n := calls
		mu.Unlock()
		res := testResult(n)
		if n == 1 {
			// fallback: the entire week is shifted back, does not cover today
			for i := range res.Schedule.Days {
				res.Schedule.Days[i].Date = lastWeek.AddDate(0, 0, i)
			}
		}
		return res, nil
	})
	srv.cfg.CacheTTL = time.Hour

	rr := get(t, srv, "GET", "/schedule/current", nil)
	if rr.Code != http.StatusOK {
		t.Fatalf("first: code=%d", rr.Code)
	}
	advance() // stale now
	rr = get(t, srv, "GET", "/schedule/current", nil)
	if rr.Code != http.StatusOK {
		t.Fatalf("second: code=%d body=%s", rr.Code, rr.Body.String())
	}
	if rr.Header().Get("X-Served-From") != "cache" {
		t.Fatal("a stale non-covering entry must be served from the cache, not refetched")
	}
	if !strings.Contains(rr.Body.String(), "ΦΑΡΜΑΚΕΙΟ 1") {
		t.Fatal("the fallback (best available) week must be served")
	}
	mu.Lock()
	if calls != 1 {
		t.Fatalf("expected no request-path refetch, got %d", calls)
	}
	mu.Unlock()
}

func TestJSONShape(t *testing.T) {
	res := testResult(1)
	jsonBytes, err := res.JSON()
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	if err := json.Unmarshal(jsonBytes, &doc); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"schedule", "image_sha256", "timings"} {
		if _, ok := doc[key]; !ok {
			t.Fatalf("missing key %q in %s", key, jsonBytes)
		}
	}
}
