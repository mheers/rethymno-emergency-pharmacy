// Command rethymno-emergency-pharmacy is the CLI for the Rethymno pharmacy duty-schedule
// OCR pipeline.
//
// Usage:
//
//	rethymno-emergency-pharmacy ingest                    download the current FSKriti schedule and parse it
//	rethymno-emergency-pharmacy parse [flags] ./schedule.jpg    parse a local schedule image to JSON
//	rethymno-emergency-pharmacy inspect ./schedule.jpg    print geometry/columns/OCR diagnostics
//	rethymno-emergency-pharmacy benchmark ./testdata/schedules
//	rethymno-emergency-pharmacy serve [flags]             internal HTTP API with a daily-refreshed cache
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	rethymnoemergency "github.com/mheers/rethymno-emergency-pharmacy"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/adjudicate"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/extract"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/fetch"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/ocr"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/pipeline"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/server"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/vision"
)

var version = "1.0.0"

func main() {
	log.SetFlags(log.LstdFlags)
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	var err error
	switch os.Args[1] {
	case "ingest":
		err = cmdIngest(ctx, os.Args[2:])
	case "parse":
		err = cmdParse(ctx, os.Args[2:])
	case "inspect":
		err = cmdInspect(ctx, os.Args[2:])
	case "benchmark":
		err = cmdBenchmark(ctx, os.Args[2:])
	case "serve":
		err = cmdServe(ctx, os.Args[2:])
	case "ocr-worker":
		err = rethymnoemergency.WorkerMain(os.Args[2:])
	case "version", "-v", "--version":
		fmt.Printf("rethymno-emergency-pharmacy %s\n", version)
		return
	case "help", "-h", "--help":
		usage()
		return
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `rethymno-emergency-pharmacy - local OCR pipeline for the Rethymno pharmacy duty schedules

Usage:
  rethymno-emergency-pharmacy ingest                     download current FSKriti schedule, parse, print JSON
  rethymno-emergency-pharmacy parse [flags] <image>      parse a local schedule image to JSON
  rethymno-emergency-pharmacy inspect <image> [flags]    print preprocessing + OCR diagnostics
  rethymno-emergency-pharmacy benchmark <dir> [flags]    run stage timings over images in <dir>
  rethymno-emergency-pharmacy serve [flags]              serve the internal HTTP API (24h cache)
  rethymno-emergency-pharmacy version                    print version

Flags:
  --models <dir>    ONNX model directory (default: embedded in the binary)
  --debug <dir>     write debugging artifacts under <dir>
  --source <url>    source URL recorded in the JSON
  --city <name>     city name recorded in the JSON (default: Ρέθυμνο)
  --iterations <n>  benchmark repetitions (default 1)
  --listen <addr>   serve: HTTP listen address (default 127.0.0.1:8080)
  --cache-ttl <d>   serve: cache TTL for parsed schedules (default 24h)
  --judge           ingest/parse/serve: adjudicate ambiguous catalog matches with
                    TypeSafe System One (sends OCR text to api.typesafe.ai;
                    requires TYPESAFE_API_KEY; decisions are cached)
  --judge-model     pinned System One model (default jev-1.13.0)
  --judge-cache     identity decision cache file (default: user cache dir)
`)
}

// judgeFlags groups the identity-adjudication flags shared by ingest, parse
// and serve. Judging is opt-in and default off: it sends OCR text (public
// pharmacy data) to api.typesafe.ai, unlike the local, CPU-only pipeline.
type judgeFlags struct {
	enable *bool
	model  *string
	cache  *string
}

func addJudgeFlags(fs *flag.FlagSet) *judgeFlags {
	return &judgeFlags{
		enable: fs.Bool("judge", false, "adjudicate ambiguous catalog matches with TypeSafe System One"),
		model:  fs.String("judge-model", adjudicate.DefaultJudgeModel, "pinned System One model"),
		cache:  fs.String("judge-cache", defaultJudgeCachePath(), "identity decision cache file"),
	}
}

func (j *judgeFlags) config() *rethymnoemergency.IdentityJudgeConfig {
	if j == nil || !*j.enable {
		return nil
	}
	return &rethymnoemergency.IdentityJudgeConfig{
		Model:     *j.model,
		CachePath: *j.cache,
	}
}

func defaultJudgeCachePath() string {
	dir, err := os.UserCacheDir()
	if err != nil {
		return ""
	}
	return filepath.Join(dir, "rethymno-emergency-pharmacy", "identity-decisions.json")
}

// openEngine builds the vision processor (in-process, gocv) and the OCR
// backend (dedicated worker subprocess, ORT only). The two must not share
// a process: loading the ONNX Runtime library corrupts OpenCV allocations.
func openEngine(modelsDir string, threads int) (*vision.Processor, pipeline.OCRBackend, error) {
	v := vision.New(vision.Config{
		TargetDPI:         150,
		MaxWidth:          4096,
		AdaptiveBlockSize: 31,
		AdaptiveC:         10,
		Deskew:            true,
		RemoveGridLines:   true,
		DetectColumns:     true,
	})
	exe, err := os.Executable()
	if err != nil {
		return nil, nil, fmt.Errorf("executable path: %w", err)
	}
	remote, err := ocr.StartRemote([]string{exe, "ocr-worker"}, modelsDir, "", threads)
	if err != nil {
		return nil, nil, fmt.Errorf("ocr worker: %w", err)
	}
	return v, remote, nil
}

func cmdParse(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("parse", flag.ExitOnError)
	modelsDir := fs.String("models", "", "ONNX model directory (empty = embedded)")
	recModel := fs.String("rec-model", "small", "embedded recognition model: small, medium or tiny")
	debugDir := fs.String("debug", "", "debug output directory")
	source := fs.String("source", "", "source URL recorded in JSON")
	city := fs.String("city", "Ρέθυμνο", "city recorded in JSON")
	judge := addJudgeFlags(fs)
	fs.Parse(args)
	if fs.NArg() < 1 {
		return fmt.Errorf("usage: rethymno-emergency-pharmacy parse <image> [flags]")
	}
	client, err := rethymnoemergency.New(rethymnoemergency.ClientConfig{
		ModelPath:     *modelsDir,
		RecModel:      *recModel,
		IdentityJudge: judge.config(),
		Log:           log.Default(),
	})
	if err != nil {
		return err
	}
	defer client.Close()
	res, err := client.ParseFile(ctx, fs.Arg(0), rethymnoemergency.Options{
		SourceURL: *source,
		City:      *city,
		DebugDir:  *debugDir,
	})
	if err != nil {
		return err
	}
	out, err := res.JSON()
	if err != nil {
		return err
	}
	fmt.Println(string(out))
	return nil
}

func cmdInspect(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("inspect", flag.ExitOnError)
	modelsDir := fs.String("models", "models", "ONNX model directory")
	fs.Parse(args)
	if fs.NArg() < 1 {
		return fmt.Errorf("usage: rethymno-emergency-pharmacy inspect <image>")
	}
	data, err := os.ReadFile(fs.Arg(0))
	if err != nil {
		return err
	}
	v, eng, err := openEngine(*modelsDir, 0)
	if err != nil {
		return err
	}
	defer eng.Close()

	img, err := v.Decode(data)
	if err != nil {
		return err
	}
	defer img.Close()
	fmt.Printf("image geometry: %dx%d, %d channels, aspect %.3f\n",
		img.Cols(), img.Rows(), img.Channels(), float64(img.Cols())/float64(img.Rows()))

	proc, err := v.Process(img)
	if err != nil {
		return err
	}
	defer pipeline.CloseProcessed(proc)
	g := proc.Geometry
	fmt.Printf("content rect:   %v (perspective corrected=%v, rotation=%.2f deg)\n",
		g.ContentRect, g.PerspectiveCorrected, g.RotationDegrees)
	fmt.Printf("detected columns: %d\n", len(proc.Columns))
	for _, c := range proc.Columns {
		fmt.Printf("  col %d: x %d..%d width %d\n", c.Index+1, c.X0, c.X1, c.Width)
	}

	pipe := pipeline.New(v, eng, nil)
	lines, err := pipe.OCRColumnLines(proc)
	if err != nil {
		return err
	}
	fmt.Printf("OCR line count: %d\n", len(lines))
	var total float32
	for _, l := range lines {
		total += l.Confidence
	}
	if len(lines) > 0 {
		fmt.Printf("average OCR confidence: %.3f\n", total/float32(len(lines)))
	}
	byCol := make([]int, len(proc.Columns))
	for _, l := range lines {
		cx := (l.Box.Min.X + l.Box.Max.X) / 2
		for i, c := range proc.Columns {
			if cx >= c.Rect.Min.X && cx <= c.Rect.Max.X {
				byCol[i]++
			}
		}
	}
	for i, n := range byCol {
		fmt.Printf("  col %d lines: %d\n", i+1, n)
	}
	for _, l := range lines {
		fmt.Printf("  [y=%4d x=%4d c=%.2f] %s\n", l.Box.Min.Y, l.Box.Min.X, l.Confidence, l.Text)
	}
	return nil
}

func cmdIngest(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("ingest", flag.ExitOnError)
	modelsDir := fs.String("models", "", "ONNX model directory (empty = embedded)")
	outJSON := fs.String("out", "", "write JSON to this file instead of stdout")
	judge := addJudgeFlags(fs)
	fs.Parse(args)

	client := fetch.New(30 * time.Second)
	fmt.Fprintln(os.Stderr, "fetching FSKriti page ...")
	page, err := client.Get(ctx, extract.FSKritiPageURL)
	if err != nil {
		return fmt.Errorf("fetch FSKriti: %w", err)
	}
	imgs := extract.FindScheduleImages(page)
	fmt.Fprintf(os.Stderr, "found %d schedule images\n", len(imgs))
	for _, im := range imgs {
		fmt.Fprintf(os.Stderr, "  %s (%s .. %s)\n", im.URL, im.From.Format("02/01/2006"), im.To.Format("02/01/2006"))
	}
	cur, err := extract.SelectCurrent(imgs, time.Now())
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "downloading %s ...\n", cur.URL)

	ocrClient, err := rethymnoemergency.New(rethymnoemergency.ClientConfig{
		ModelPath:     *modelsDir,
		IdentityJudge: judge.config(),
		Log:           log.Default(),
	})
	if err != nil {
		return err
	}
	defer ocrClient.Close()
	res, err := ocrClient.ParseURL(ctx, cur.URL, rethymnoemergency.Options{City: "Ρέθυμνο", DebugDir: "debug"})
	if err != nil {
		return err
	}
	out, err := res.JSON()
	if err != nil {
		return err
	}
	if *outJSON != "" {
		if err := os.WriteFile(*outJSON, out, 0o644); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "wrote %s\n", *outJSON)
	} else {
		fmt.Println(string(out))
	}
	for _, w := range res.Warnings {
		fmt.Fprintf(os.Stderr, "warning: %s\n", w)
	}
	return nil
}

func cmdBenchmark(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("benchmark", flag.ExitOnError)
	modelsDir := fs.String("models", "models", "ONNX model directory")
	iterations := fs.Int("iterations", 1, "runs per image")
	fs.Parse(args)
	dir := fs.Arg(0)
	if dir == "" {
		dir = "testdata/schedules"
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	var files []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(strings.ToLower(e.Name()), ".jpg") {
			files = append(files, filepath.Join(dir, e.Name()))
		}
	}
	if len(files) == 0 {
		return fmt.Errorf("no jpg files in %s", dir)
	}
	v, eng, err := openEngine(*modelsDir, 0)
	if err != nil {
		return err
	}
	defer eng.Close()
	pipe := pipeline.New(v, eng, nil)

	var totals pipeline.StageTimings
	count := 0
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			return err
		}
		for i := 0; i < *iterations; i++ {
			img, err := v.Decode(data)
			if err != nil {
				return err
			}
			res, err := pipe.Run(img, pipeline.Options{})
			img.Close()
			if err != nil {
				return err
			}
			totals.Decode += res.Timings.Decode
			totals.Perspective += res.Timings.Perspective
			totals.Deskew += res.Timings.Deskew
			totals.Grid += res.Timings.Grid
			totals.Columns += res.Timings.Columns
			totals.OCRA += res.Timings.OCRA
			totals.Layout += res.Timings.Layout
			totals.Parse += res.Timings.Parse
			totals.Validate += res.Timings.Validate
			totals.Total += res.Timings.Total
			count++
		}
	}
	n := float64(count)
	fmt.Printf("Stage                  Time\n")
	fmt.Printf("--------------------------------\n")
	fmt.Printf("%-21s %8.1fms\n", "Decode", float64(totals.Decode.Milliseconds())/n)
	fmt.Printf("%-21s %8.1fms\n", "Vision (persp+grid+cols)", float64(totals.Perspective.Milliseconds())/n)
	fmt.Printf("%-21s %8.1fms\n", "OCR (det+rec)", float64(totals.OCRA.Milliseconds())/n)
	fmt.Printf("%-21s %8.1fms\n", "Layout", float64(totals.Layout.Milliseconds())/n)
	fmt.Printf("%-21s %8.1fms\n", "Parse", float64(totals.Parse.Milliseconds())/n)
	fmt.Printf("%-21s %8.1fms\n", "Validate", float64(totals.Validate.Milliseconds())/n)
	fmt.Printf("--------------------------------\n")
	fmt.Printf("%-21s %8.1fms\n", "Total", float64(totals.Total.Milliseconds())/n)
	fmt.Printf("(%d runs across %d images)\n", count, len(files))
	return nil
}

// cmdServe runs the internal HTTP API: fetches the current schedule image
// once a day at local midnight, caches the parsed result (default TTL 24h),
// and serves requests exclusively from the cache so the OCR pipeline runs
// at most once per day.
func cmdServe(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	listen := fs.String("listen", "127.0.0.1:8080", "HTTP listen address")
	cacheTTL := fs.Duration("cache-ttl", 24*time.Hour, "cache TTL for parsed schedules")
	modelsDir := fs.String("models", "", "ONNX model directory (empty = embedded)")
	recModel := fs.String("rec-model", "small", "embedded recognition model: small, medium or tiny")
	city := fs.String("city", "Ρέθυμνο", "city recorded in JSON")
	source := fs.String("source", "", "source URL recorded in JSON")
	judge := addJudgeFlags(fs)
	fs.Parse(args)
	if fs.NArg() > 0 {
		return fmt.Errorf("usage: rethymno-emergency-pharmacy serve [flags]")
	}

	client, err := rethymnoemergency.New(rethymnoemergency.ClientConfig{
		ModelPath:     *modelsDir,
		RecModel:      *recModel,
		IdentityJudge: judge.config(),
		Log:           log.Default(),
	})
	if err != nil {
		return err
	}
	defer client.Close()

	srv := server.New(server.Config{
		Client:    client,
		Listen:    *listen,
		CacheTTL:  *cacheTTL,
		City:      *city,
		SourceURL: *source,
		Log:       log.Default(),
	})
	return srv.Run(ctx)
}
