// Package rethymnoemergency is the public library API for the pharmacy
// duty-schedule OCR pipeline (fetch -> vision -> OCR -> layout -> parse ->
// validate).
//
// The ONNX Runtime library and the PP-OCRv6 models are embedded into the
// binary (see assets_embed.go), so a program importing this package needs
// no model files on disk — only the OpenCV shared libraries that gocv
// links against.
//
// Minimal use:
//
//	client, err := rethymnoemergency.New(rethymnoemergency.ClientConfig{})
//	if err != nil { log.Fatal(err) }
//	defer client.Close()
//	out, err := client.ParseJSON(ctx, "schedule.jpg", rethymnoemergency.Options{})
//
// The default OCR backend runs inference in a worker subprocess (the
// ONNX Runtime library corrupts OpenCV allocations when loaded into the
// same process; see internal/ocr). The host binary must wire up the
// worker subcommand for this to work:
//
//	func main() {
//		if len(os.Args) > 1 && os.Args[1] == "ocr-worker" {
//			os.Exit(rethymnoemergency.WorkerMain(os.Args[2:]))
//		}
//		// ... application code ...
//	}
//
// Alternatively set ClientConfig{Backend: BackendInProcess} to run
// inference in-process, or ClientConfig{WorkerCommand} to point at a
// binary that does.
package rethymnoemergency

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/mheers/rethymno-emergency-pharmacy/internal/adjudicate"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/fetch"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/ocr"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/pipeline"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/validate"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/vision"
	jev "github.com/mheers/typesafeai-systemone-jev-go"
)

// Options configures a pipeline run.
type Options = pipeline.Options

// Result is the full pipeline output, including validation against the
// embedded pharmacy catalog. Use Result.JSON to serialize it.
type Result = pipeline.Result

// Backend selects how the OCR stage runs.
type Backend int

const (
	// BackendSubprocess (default) runs OCR inference in a worker
	// subprocess. The host binary must wire up WorkerMain, or set
	// ClientConfig.WorkerCommand.
	BackendSubprocess Backend = iota
	// BackendInProcess runs OCR inference in this process. Loading ONNX
	// Runtime and OpenCV in one process is known to corrupt OpenCV
	// allocations on some glibc versions; test before relying on it.
	BackendInProcess
)

// ClientConfig configures a Client.
type ClientConfig struct {
	// ModelPath is an ONNX models directory on disk
	// (PP-OCRv6_*_onnx/inference.onnx). Empty selects the embedded
	// models.
	ModelPath string
	// RecModel selects the embedded recognition model: "small"
	// (default), "medium" or "tiny". Ignored when ModelPath is set.
	RecModel string
	// Backend selects the OCR backend (default BackendSubprocess).
	Backend Backend
	// WorkerCommand overrides the worker process command (default:
	// this executable with the "ocr-worker" subcommand).
	WorkerCommand []string
	// NumThreads sets ONNX inference threads (0 = engine default).
	NumThreads int
	// HTTPTimeout bounds URL downloads (default 30s).
	HTTPTimeout time.Duration
	// IdentityJudge enables TypeSafe System One catalog-identity adjudication
	// for pharmacies whose phone number matches several catalog entries. It
	// sends OCR text (public pharmacy data) to api.typesafe.ai, so it is
	// opt-in and breaks the otherwise local, CPU-only default. Nil keeps the
	// deterministic similarity pick. TYPESAFE_API_KEY must be set.
	IdentityJudge *IdentityJudgeConfig
	// PlausibilityJudge enables TypeSafe System One plausibility verification
	// for entries the catalog cannot rescue (their phone number matches no
	// catalog entry): one Noul per field, consumed as additive warnings only
	// (TYPESAFE_EVALUATION.md §3D). It sends the same class of OCR text and is
	// just as opt-in as IdentityJudge. Nil disables it and keeps the
	// deterministic path.
	PlausibilityJudge *PlausibilityJudgeConfig
	// Log receives pipeline diagnostics (default stderr logger).
	Log *log.Logger
}

// IdentityJudgeConfig configures runtime catalog-identity adjudication
// (TYPESAFE_EVALUATION.md §3A). Every decision is recorded in CachePath and
// reused, so repeated parses produce identical output; a group whose decision
// is missing is evaluated once and persisted.
type IdentityJudgeConfig struct {
	// Model is the pinned System One model. Empty selects the model the
	// 0.8 gates were measured with (adjudicate.DefaultJudgeModel).
	Model string
	// CachePath is the decision cache file. Required: persistence is what
	// keeps the JSON deterministic across runs.
	CachePath string
	// MinConfidence gates the Choice answer's confidence; zero selects the
	// measured default (0.8).
	MinConfidence float64
	// MinSamePharmacy gates the same-pharmacy Noul; zero selects the
	// measured default (0.8).
	MinSamePharmacy float64
}

// PlausibilityJudgeConfig configures runtime plausibility verification
// (TYPESAFE_EVALUATION.md §3D). Every decision is recorded in CachePath and
// reused, so repeated parses produce identical warnings; the gates are the
// measured ones (address 0.7, name 0.5) and each flag is a warning, never a
// rewrite.
type PlausibilityJudgeConfig struct {
	// Model is the pinned System One model. Empty selects
	// adjudicate.DefaultJudgeModel.
	Model string
	// CachePath is the decision cache file. Required: persistence is what
	// keeps the JSON deterministic across runs. It may be the same path as
	// IdentityJudge.CachePath; the two judges then share one file and one
	// in-memory cache.
	CachePath string
	// MinName gates the name-plausibility Noul; zero selects the measured
	// default (0.5).
	MinName float64
	// MinAddress gates the address-plausibility Noul; zero selects the
	// measured default (0.7).
	MinAddress float64
}

const (
	defaultIdentityGate            = 0.8
	defaultPlausibilityNameGate    = 0.5
	defaultPlausibilityAddressGate = 0.7
)

// Client runs the OCR pipeline.
type Client struct {
	cfg        ClientConfig
	pipe       *pipeline.Pipeline
	backend    pipeline.OCRBackend
	judge      pipeline.IdentityJudge
	judgeCfg   adjudicate.IdentityConfig
	plausJudge pipeline.PlausibilityJudge
	plausCfg   adjudicate.PlausibilityConfig
}

// New creates a Client and initializes the vision processor and the OCR
// backend.
func New(cfg ClientConfig) (*Client, error) {
	if cfg.ModelPath == "" && len(embeddedRecModelSmall) == 0 {
		return nil, errors.New("rethymnoemergency: no models available: run scripts/bootstrap.sh, or set ClientConfig.ModelPath")
	}
	if cfg.Log == nil {
		cfg.Log = log.New(os.Stderr, "[rethymnoemergency] ", log.LstdFlags)
	}
	if cfg.HTTPTimeout == 0 {
		cfg.HTTPTimeout = 30 * time.Second
	}
	if _, err := recModelBytes(cfg.RecModel); err != nil {
		return nil, err
	}
	if cfg.ModelPath == "" {
		rec, err := recModelBytes(cfg.RecModel)
		if err != nil {
			return nil, err
		}
		ocr.SetEmbeddedLib(embeddedORTLib)
		ocr.SetEmbeddedModels(embeddedDetModel, rec)
	}

	// The judges are configured before the OCR backend so that a
	// configuration error cannot leave a worker process behind.
	caches := &judgeCaches{byPath: map[string]*adjudicate.DecisionCache{}}
	judge, judgeCfg, err := buildIdentityJudge(cfg.IdentityJudge, caches.open)
	if err != nil {
		return nil, err
	}
	plausJudge, plausCfg, err := buildPlausibilityJudge(cfg.PlausibilityJudge, caches.open)
	if err != nil {
		return nil, err
	}

	v := vision.New(vision.Config{
		TargetDPI:         150,
		MaxWidth:          4096,
		AdaptiveBlockSize: 31,
		AdaptiveC:         10,
		Deskew:            true,
		RemoveGridLines:   true,
		DetectColumns:     true,
	})

	var backend pipeline.OCRBackend
	switch cfg.Backend {
	case BackendInProcess:
		backend, err = ocr.New(ocr.Config{
			ModelPath:   cfg.ModelPath,
			NoDetection: true,
			NumThreads:  cfg.NumThreads,
			Log:         cfg.Log,
		})
	default:
		cmd := cfg.WorkerCommand
		if len(cmd) == 0 {
			exe, e := os.Executable()
			if e != nil {
				return nil, fmt.Errorf("rethymnoemergency: executable path: %w", e)
			}
			cmd = []string{exe, "ocr-worker"}
		}
		backend, err = ocr.StartRemote(cmd, cfg.ModelPath, cfg.RecModel, cfg.NumThreads)
	}
	if err != nil {
		return nil, fmt.Errorf("rethymnoemergency: ocr backend: %w", err)
	}

	return &Client{
		cfg:        cfg,
		pipe:       pipeline.New(v, backend, cfg.Log),
		backend:    backend,
		judge:      judge,
		judgeCfg:   judgeCfg,
		plausJudge: plausJudge,
		plausCfg:   plausCfg,
	}, nil
}

// judgeCaches hands out one DecisionCache per path. Identity and plausibility
// decisions share a cache file when configured with the same path, and sharing
// the in-memory instance is what keeps each judge from overwriting the other's
// decisions when it saves.
type judgeCaches struct {
	byPath map[string]*adjudicate.DecisionCache
}

func (c *judgeCaches) open(path string) (*adjudicate.DecisionCache, error) {
	if cache, ok := c.byPath[path]; ok {
		return cache, nil
	}
	cache, err := adjudicate.OpenDecisionCache(path)
	if err != nil {
		return nil, err
	}
	c.byPath[path] = cache
	return cache, nil
}

// buildIdentityJudge wires the optional runtime adjudicator. Any
// configuration error is returned before the OCR backend is created.
func buildIdentityJudge(cfg *IdentityJudgeConfig, openCache func(string) (*adjudicate.DecisionCache, error)) (pipeline.IdentityJudge, adjudicate.IdentityConfig, error) {
	if cfg == nil {
		return nil, adjudicate.IdentityConfig{}, nil
	}
	if strings.TrimSpace(cfg.CachePath) == "" {
		return nil, adjudicate.IdentityConfig{}, errors.New("rethymnoemergency: IdentityJudge.CachePath is required: decisions must be recorded to keep the JSON deterministic")
	}
	key := os.Getenv(adjudicate.APIKeyEnv)
	if key == "" {
		return nil, adjudicate.IdentityConfig{}, fmt.Errorf("rethymnoemergency: identity adjudication needs %s", adjudicate.APIKeyEnv)
	}
	model := cfg.Model
	if model == "" {
		model = adjudicate.DefaultJudgeModel
	}
	client, err := adjudicate.NewClient(key, jev.WithModel(model))
	if err != nil {
		return nil, adjudicate.IdentityConfig{}, fmt.Errorf("rethymnoemergency: identity adjudication client: %w", err)
	}
	cache, err := openCache(cfg.CachePath)
	if err != nil {
		return nil, adjudicate.IdentityConfig{}, fmt.Errorf("rethymnoemergency: identity decision cache: %w", err)
	}
	minConf, minSame := cfg.MinConfidence, cfg.MinSamePharmacy
	if minConf == 0 {
		minConf = defaultIdentityGate
	}
	if minSame == 0 {
		minSame = defaultIdentityGate
	}
	return adjudicate.NewCachedJudge(client, cache), adjudicate.IdentityConfig{
		MinConfidence:   minConf,
		MinSamePharmacy: minSame,
	}, nil
}

// buildPlausibilityJudge wires the optional runtime plausibility verifier.
// Any configuration error is returned before the OCR backend is created.
func buildPlausibilityJudge(cfg *PlausibilityJudgeConfig, openCache func(string) (*adjudicate.DecisionCache, error)) (pipeline.PlausibilityJudge, adjudicate.PlausibilityConfig, error) {
	if cfg == nil {
		return nil, adjudicate.PlausibilityConfig{}, nil
	}
	if strings.TrimSpace(cfg.CachePath) == "" {
		return nil, adjudicate.PlausibilityConfig{}, errors.New("rethymnoemergency: PlausibilityJudge.CachePath is required: decisions must be recorded to keep the JSON deterministic")
	}
	key := os.Getenv(adjudicate.APIKeyEnv)
	if key == "" {
		return nil, adjudicate.PlausibilityConfig{}, fmt.Errorf("rethymnoemergency: plausibility verification needs %s", adjudicate.APIKeyEnv)
	}
	model := cfg.Model
	if model == "" {
		model = adjudicate.DefaultJudgeModel
	}
	client, err := adjudicate.NewClient(key, jev.WithModel(model))
	if err != nil {
		return nil, adjudicate.PlausibilityConfig{}, fmt.Errorf("rethymnoemergency: plausibility adjudication client: %w", err)
	}
	cache, err := openCache(cfg.CachePath)
	if err != nil {
		return nil, adjudicate.PlausibilityConfig{}, fmt.Errorf("rethymnoemergency: plausibility decision cache: %w", err)
	}
	minName, minAddress := cfg.MinName, cfg.MinAddress
	if minName == 0 {
		minName = defaultPlausibilityNameGate
	}
	if minAddress == 0 {
		minAddress = defaultPlausibilityAddressGate
	}
	return adjudicate.NewCachedPlausibilityJudge(client, cache), adjudicate.PlausibilityConfig{
		MinName:    minName,
		MinAddress: minAddress,
	}, nil
}

func recModelBytes(name string) ([]byte, error) {
	switch name {
	case "", "small":
		return embeddedRecModelSmall, nil
	case "medium":
		return embeddedRecModelMedium, nil
	case "tiny":
		return embeddedRecModelTiny, nil
	}
	return nil, fmt.Errorf("rethymnoemergency: unknown rec model %q (want small, medium or tiny)", name)
}

// Parse runs the pipeline on already-fetched image bytes and returns the
// result enriched with catalog lookups and validation.
func (c *Client) Parse(ctx context.Context, data []byte, opts Options) (*Result, error) {
	res, err := c.pipe.RunBytes(ctx, data, opts)
	if err != nil {
		return nil, err
	}
	references := loadReferences(c.cfg.Log)
	fill := pipeline.FillFromCatalogJudged(ctx, &res.Schedule, references, c.judge, c.judgeCfg)
	vals, warnings := pipeline.ValidateResultWithPicks(&res.Schedule, references, fill.Picks)
	res.Validations = vals
	if len(fill.Warnings) > 0 {
		warnings = append(fill.Warnings, warnings...)
	}
	res.Warnings = warnings
	res.Warnings = append(res.Warnings, pipeline.VerifyPlausibility(ctx, &res.Schedule, references, c.plausJudge, c.plausCfg)...)
	pipeline.ApplyValidation(&res.Schedule, vals)
	return res, nil
}

// ParseSource reads the image from an http(s) URL or a local file path
// (auto-detected) and returns the parsed result.
func (c *Client) ParseSource(ctx context.Context, source string, opts Options) (*Result, error) {
	if strings.HasPrefix(source, "http://") || strings.HasPrefix(source, "https://") {
		return c.ParseURL(ctx, source, opts)
	}
	return c.ParseFile(ctx, source, opts)
}

// ParseFile reads a local image file and parses it.
func (c *Client) ParseFile(ctx context.Context, path string, opts Options) (*Result, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("rethymnoemergency: read %s: %w", path, err)
	}
	if opts.SourceURL == "" {
		opts.SourceURL = path
	}
	return c.Parse(ctx, data, opts)
}

// ParseURL downloads an image over HTTP and parses it.
func (c *Client) ParseURL(ctx context.Context, url string, opts Options) (*Result, error) {
	client := fetch.New(c.cfg.HTTPTimeout)
	data, err := client.Get(ctx, url)
	if err != nil {
		return nil, fmt.Errorf("rethymnoemergency: fetch %s: %w", url, err)
	}
	if opts.SourceURL == "" {
		opts.SourceURL = url
	}
	return c.Parse(ctx, data, opts)
}

// ParseJSON is ParseSource returning the result marshaled as JSON.
func (c *Client) ParseJSON(ctx context.Context, source string, opts Options) ([]byte, error) {
	res, err := c.ParseSource(ctx, source, opts)
	if err != nil {
		return nil, err
	}
	return res.JSON()
}

// CatalogJSON returns the embedded golden pharmacy catalog (all reference
// entries, each with Google-corrected lat/lon and optional Places enrichment)
// as JSON. It is deterministic and independent of any pipeline state, so a
// consumer can cache and diff it across days.
func CatalogJSON() ([]byte, error) {
	refs, err := validate.LoadReference(validate.ReferenceJSON)
	if err != nil {
		return nil, fmt.Errorf("rethymnoemergency: catalog: %w", err)
	}
	return json.MarshalIndent(refs, "", "  ")
}

// Close shuts down the OCR backend.
func (c *Client) Close() error {
	if c.backend != nil {
		return c.backend.Close()
	}
	return nil
}

func loadReferences(log *log.Logger) []validate.Reference {
	refs, err := validate.LoadReference(validate.ReferenceJSON)
	if err != nil {
		log.Printf("warning: reference catalog: %v", err)
		return nil
	}
	return refs
}

// WorkerMain serves OCR inference requests on stdin/stdout until stdin
// closes. Wire it into the host binary behind an "ocr-worker" subcommand
// so the subprocess backend can spawn itself.
func WorkerMain(args []string) error {
	fs := flag.NewFlagSet("ocr-worker", flag.ExitOnError)
	modelsDir := fs.String("models", "", "ONNX model directory (empty = embedded)")
	recModel := fs.String("rec-model", "small", "embedded recognition model: small, medium or tiny")
	threads := fs.Int("threads", 0, "inference threads (0 = auto)")
	fs.Parse(args)
	rec, err := recModelBytes(*recModel)
	if err != nil {
		return err
	}
	ocr.SetEmbeddedLib(embeddedORTLib)
	ocr.SetEmbeddedModels(embeddedDetModel, rec)
	return ocr.RunWorker(*modelsDir, *threads)
}
