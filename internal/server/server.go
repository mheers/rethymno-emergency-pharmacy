package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	rethymnoemergency "github.com/mheers/rethymno-emergency-pharmacy"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/extract"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/fetch"
)

// Config configures a Server.
type Config struct {
	Client    *rethymnoemergency.Client
	Fetch     *fetch.Client
	Listen    string
	CacheTTL  time.Duration
	City      string
	SourceURL string
	Log       *log.Logger
}

// Server is the internal HTTP API for the OCR pipeline.
type Server struct {
	cfg    Config
	client *rethymnoemergency.Client
	fetch  *fetch.Client
	cache  *cache
	log    *log.Logger
	now    func() time.Time
	// nextRefresh is the next scheduled refresh time, maintained by
	// refreshLoop/refreshNow.
	nextRefresh time.Time

	fmu     sync.Mutex
	flights map[string]*flight

	// fetchParse is the fetch+parse unit for one schedule week (zero date =
	// current week); tests override it.
	fetchParse func(ctx context.Context, date time.Time) (*rethymnoemergency.Result, error)
	// parseImage parses caller-supplied image bytes; tests override it.
	parseImage func(ctx context.Context, data []byte, opts rethymnoemergency.Options) (*rethymnoemergency.Result, error)
}

// New creates a Server. The client is optional only for tests; production
// use always passes a rethymnoemergency.Client.
func New(cfg Config) *Server {
	if cfg.Fetch == nil {
		cfg.Fetch = fetch.New(30 * time.Second)
	}
	if cfg.Listen == "" {
		cfg.Listen = "127.0.0.1:8080"
	}
	if cfg.CacheTTL <= 0 {
		cfg.CacheTTL = 24 * time.Hour
	}
	if cfg.Log == nil {
		cfg.Log = log.New(os.Stderr, "[serve] ", log.LstdFlags)
	}
	s := &Server{
		cfg:     cfg,
		client:  cfg.Client,
		fetch:   cfg.Fetch,
		cache:   newCache(time.Now),
		log:     cfg.Log,
		now:     time.Now,
		flights: map[string]*flight{},
	}
	s.fetchParse = s.parseWeek
	s.parseImage = func(ctx context.Context, data []byte, opts rethymnoemergency.Options) (*rethymnoemergency.Result, error) {
		if s.client == nil {
			return nil, errors.New("no rethymnoemergency client configured")
		}
		return s.client.Parse(ctx, data, opts)
	}
	return s
}

// Run serves HTTP and runs the daily midnight refresh loop until ctx is
// cancelled.
func (s *Server) Run(ctx context.Context) error {
	go s.refreshLoop(ctx)
	return s.Serve(ctx)
}

// Serve serves HTTP until ctx is cancelled or the listener fails.
func (s *Server) Serve(ctx context.Context) error {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("GET /schedule/current", s.handleCurrent)
	mux.HandleFunc("GET /schedule/week", s.handleWeek)
	mux.HandleFunc("GET /schedule/image", s.handleImage)
	mux.HandleFunc("POST /parse", s.handleParse)

	httpServer := &http.Server{
		Addr:              s.cfg.Listen,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}
	s.log.Printf("listening on http://%s (cache ttl %s)", s.cfg.Listen, s.cfg.CacheTTL)

	errc := make(chan error, 1)
	go func() { errc <- httpServer.ListenAndServe() }()
	select {
	case <-ctx.Done():
		shCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return httpServer.Shutdown(shCtx)
	case err := <-errc:
		return err
	}
}

// flight deduplicates concurrent fetches of the same cache key so the
// upstream is hit at most once per week per refresh.
type flight struct {
	done chan struct{}
	res  *rethymnoemergency.Result
	err  error
}

// entryFor returns the cached entry for the week of date (zero = current).
//
// Fresh entries are never refetched per request — freshness is the daily
// refresh's job (see refresh.go), so the upstream is hit at most once a
// day. Stale entries are revalidated in the background so a slow upstream
// never blocks a client that already has a result — unless the stale entry
// provably does not cover the requested date, in which case the fetch
// blocks and the caller receives the correct week instead of a known-wrong
// one.
func (s *Server) entryFor(ctx context.Context, date time.Time) (*entry, bool, error) {
	ref := weekAnchor(date, s.now())
	key := weekKey(ref)
	if e := s.cache.get(key); e != nil {
		if coversDate(e.result, ref) {
			if s.cache.fresh(e, s.cfg.CacheTTL) {
				return e, false, nil
			}
			go s.fetchAndStore(context.WithoutCancel(ctx), key, date)
			return e, false, nil
		}
		// The entry does not cover the requested date: it is a fallback
		// (the week's image is not published yet). While fresh it is the
		// best available answer; when stale the caller must not receive a
		// known-wrong week, so fetch synchronously.
		if s.cache.fresh(e, s.cfg.CacheTTL) {
			return e, false, nil
		}
	}
	e, err := s.fetchAndStore(ctx, key, date)
	if err != nil {
		return nil, false, err
	}
	return e, true, nil
}

// coversDate reports whether the result's parsed schedule includes date.
// A fallback result (the newest image while the requested week's image is
// not yet published) covers its own week but not the requested date.
func coversDate(res *rethymnoemergency.Result, date time.Time) bool {
	if res == nil {
		return false
	}
	for i := range res.Schedule.Days {
		d := res.Schedule.Days[i].Date
		if d.Year() == date.Year() && d.Month() == date.Month() && d.Day() == date.Day() {
			return true
		}
	}
	return false
}

// fetchAndStore fetches and caches the schedule for date (zero = current),
// coalescing concurrent callers for the same key.
func (s *Server) fetchAndStore(ctx context.Context, key string, date time.Time) (*entry, error) {
	s.fmu.Lock()
	if f, ok := s.flights[key]; ok {
		s.fmu.Unlock()
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-f.done:
		}
		if f.err != nil {
			return nil, f.err
		}
		return s.newEntry(f.res), nil
	}
	f := &flight{done: make(chan struct{})}
	s.flights[key] = f
	s.fmu.Unlock()

	res, err := s.fetchParse(ctx, date)
	f.res, f.err = res, err
	close(f.done)
	if err != nil {
		s.fmu.Lock()
		delete(s.flights, key)
		s.fmu.Unlock()
		return nil, err
	}
	e := s.newEntry(res)
	// Store the entry before removing the flight so a caller that missed
	// the flight can only ever observe either the flight or the cache
	// entry — never a duplicate fetch.
	s.fmu.Lock()
	s.cache.put(key, e)
	delete(s.flights, key)
	s.fmu.Unlock()
	return e, nil
}

func (s *Server) newEntry(res *rethymnoemergency.Result) *entry {
	jsonBytes, err := res.JSON()
	if err != nil {
		jsonBytes = []byte(`{"error":"failed to marshal result"}`)
	}
	e := &entry{
		result:    res,
		json:      jsonBytes,
		etag:      res.ImageSHA256,
		fetchedAt: s.now(),
	}
	return e
}

// parseWeek downloads the schedule image covering date (zero = now) from
// fskriti.gr and parses it.
func (s *Server) parseWeek(ctx context.Context, date time.Time) (*rethymnoemergency.Result, error) {
	if s.client == nil {
		return nil, errors.New("server: no rethymnoemergency client configured")
	}
	page, err := s.fetch.Get(ctx, extract.FSKritiPageURL)
	if err != nil {
		return nil, fmt.Errorf("fetch fskriti page: %w", err)
	}
	imgs := extract.FindScheduleImages(page)
	if len(imgs) == 0 {
		return nil, errors.New("no schedule images found on fskriti page")
	}
	ref := date
	if ref.IsZero() {
		ref = s.now()
	}
	var img extract.ScheduleImage
	for _, im := range imgs {
		if !ref.Before(im.From) && !ref.After(im.To) {
			img = im
			break
		}
	}
	if img.URL == "" {
		img = imgs[len(imgs)-1] // fall back to the newest image
	}
	s.log.Printf("parsing %s (%s .. %s)",
		img.URL, img.From.Format("02/01/2006"), img.To.Format("02/01/2006"))
	return s.client.ParseURL(ctx, img.URL, rethymnoemergency.Options{
		City:      s.cfg.City,
		SourceURL: s.cfg.SourceURL,
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	io.WriteString(w, `{"ok":true}`)
}

func (s *Server) handleCurrent(w http.ResponseWriter, r *http.Request) {
	s.handleSchedule(w, r, time.Time{})
}

func (s *Server) handleWeek(w http.ResponseWriter, r *http.Request) {
	dateStr := r.URL.Query().Get("date")
	if dateStr == "" {
		writeError(w, http.StatusBadRequest, errors.New("missing ?date=DD/MM/YYYY"))
		return
	}
	date, err := parseRequestDate(dateStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	s.handleSchedule(w, r, date)
}

func (s *Server) handleSchedule(w http.ResponseWriter, r *http.Request, date time.Time) {
	e, live, err := s.entryFor(r.Context(), date)
	if err != nil {
		writeError(w, http.StatusBadGateway, err)
		return
	}
	s.writeResult(w, r, e, !live)
}

func (s *Server) handleImage(w http.ResponseWriter, r *http.Request) {
	sha := r.URL.Query().Get("sha256")
	if sha == "" {
		writeError(w, http.StatusBadRequest, errors.New("missing ?sha256="))
		return
	}
	e := s.cache.getSHA(sha)
	if e == nil {
		writeError(w, http.StatusNotFound, fmt.Errorf("no cached result for sha256 %s", sha))
		return
	}
	s.writeResult(w, r, e, true)
}

func (s *Server) handleParse(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("multipart: %w", err))
		return
	}
	file, _, err := r.FormFile("image")
	if err != nil {
		writeError(w, http.StatusBadRequest, errors.New(`missing multipart field "image"`))
		return
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, 32<<20))
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	res, err := s.parseImage(r.Context(), data, rethymnoemergency.Options{
		City:      s.cfg.City,
		SourceURL: s.cfg.SourceURL,
	})
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, fmt.Errorf("parse: %w", err))
		return
	}
	s.writeResult(w, r, s.newEntry(res), false)
}

// writeResult sends the cached JSON with ETag (image SHA-256) and
// Cache-Control headers, honoring If-None-Match with 304.
func (s *Server) writeResult(w http.ResponseWriter, r *http.Request, e *entry, fromCache bool) {
	if e.etag != "" {
		quoted := `"` + e.etag + `"`
		w.Header().Set("ETag", quoted)
		if r.Header.Get("If-None-Match") == quoted {
			w.WriteHeader(http.StatusNotModified)
			return
		}
	}
	w.Header().Set("Cache-Control", fmt.Sprintf("max-age=%d", int(s.cfg.CacheTTL.Seconds())))
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	if fromCache {
		w.Header().Set("X-Served-From", "cache")
	}
	w.Write(e.json)
}

func writeError(w http.ResponseWriter, status int, err error) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
}

// weekKey returns the cache key for the ISO week containing date.
func weekKey(date time.Time) string {
	y, w := date.ISOWeek()
	return fmt.Sprintf("week:%04d-W%02d", y, w)
}

func weekAnchor(date, now time.Time) time.Time {
	if date.IsZero() {
		return now
	}
	return date
}

// parseRequestDate accepts DD/MM/YYYY (schedule convention) or ISO dates.
func parseRequestDate(s string) (time.Time, error) {
	for _, layout := range []string{"02/01/2006", "2006-01-02"} {
		if t, err := time.ParseInLocation(layout, s, time.Local); err == nil {
			return t, nil
		}
	}
	return time.Time{}, fmt.Errorf("invalid date %q (want DD/MM/YYYY)", s)
}
