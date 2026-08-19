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

	"github.com/mheers/rethymno-emergency-pharmacy/internal/fetch"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/ocr"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/pipeline"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/validate"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/vision"
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
	// Log receives pipeline diagnostics (default stderr logger).
	Log *log.Logger
}

// Client runs the OCR pipeline.
type Client struct {
	cfg     ClientConfig
	pipe    *pipeline.Pipeline
	backend pipeline.OCRBackend
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
	var err error
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
		cfg:     cfg,
		pipe:    pipeline.New(v, backend, cfg.Log),
		backend: backend,
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
	pipeline.FillFromCatalog(&res.Schedule, references)
	vals, warnings := pipeline.ValidateResult(&res.Schedule, references)
	res.Validations = vals
	res.Warnings = warnings
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
