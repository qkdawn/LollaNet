package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

type Model struct {
	ID           string            `json:"id"`
	Name         string            `json:"name"`
	EnglishName  string            `json:"englishName"`
	Version      int               `json:"version"`
	Status       string            `json:"status"`
	Kind         string            `json:"kind"`
	Formalism    bool              `json:"formalism"`
	CategoryPath []string          `json:"categoryPath"`
	Definition   string            `json:"definition"`
	Sections     []json.RawMessage `json:"sections"`
	Sources      []json.RawMessage `json:"sources"`
	Relations    []json.RawMessage `json:"relations"`
	SourcePath   string            `json:"sourcePath"`
	Reading      *ModelReading     `json:"reading,omitempty"`
}
type ReadingSource struct {
	ID      string `json:"id"`
	Label   string `json:"label"`
	URL     string `json:"url"`
	Locator string `json:"locator"`
	Note    string `json:"note"`
}
type ModelReading struct {
	ModelID      string          `json:"modelId"`
	ModelVersion int             `json:"modelVersion"`
	Revision     int             `json:"revision"`
	Title        string          `json:"title"`
	Summary      string          `json:"summary"`
	Body         string          `json:"body"`
	Appendix     string          `json:"appendix"`
	Sources      []ReadingSource `json:"sources"`
	SourcePath   string          `json:"sourcePath"`
}
type Mode struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	EnglishName string            `json:"englishName"`
	Description string            `json:"description"`
	Color       string            `json:"color"`
	Variants    []json.RawMessage `json:"variants"`
	Sources     []json.RawMessage `json:"sources"`
}
type Library struct {
	SchemaVersion int               `json:"schemaVersion"`
	GeneratedAt   string            `json:"generatedAt"`
	Source        string            `json:"source"`
	Models        []Model           `json:"models"`
	Categories    []json.RawMessage `json:"categories"`
	Modes         []Mode            `json:"modes"`
	Associations  []json.RawMessage `json:"associations"`
}
type Filters struct{ Query, Category, Status, Kind string }
type Repository interface {
	Metadata() map[string]any
	Categories() []json.RawMessage
	Models(Filters) []Model
	Model(string) (Model, bool)
	Modes() []Mode
	Mode(string) (Mode, bool)
	Associations() []json.RawMessage
}
type JSONRepository struct{ library Library }

func LoadRepository(filename string) (*JSONRepository, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("read snapshot: %w", err)
	}
	var lib Library
	if err = json.Unmarshal(data, &lib); err != nil {
		return nil, fmt.Errorf("decode snapshot: %w", err)
	}
	if lib.SchemaVersion != 1 || lib.GeneratedAt == "" || lib.Source == "" || len(lib.Models) == 0 || len(lib.Categories) == 0 || len(lib.Modes) == 0 {
		return nil, fmt.Errorf("invalid snapshot: missing library metadata or content")
	}
	seen := map[string]bool{}
	for _, m := range lib.Models {
		if m.ID == "" || m.Name == "" || m.Version < 1 || len(m.CategoryPath) != 3 || seen[m.ID] {
			return nil, fmt.Errorf("invalid or duplicate model: %s", m.ID)
		}
		seen[m.ID] = true
		if m.Reading != nil {
			r := m.Reading
			if r.ModelID != m.ID || r.ModelVersion != m.Version || r.Revision < 1 || strings.TrimSpace(r.Title) == "" || strings.TrimSpace(r.Summary) == "" || strings.TrimSpace(r.Body) == "" || strings.TrimSpace(r.Appendix) == "" || len(r.Sources) == 0 {
				return nil, fmt.Errorf("invalid reading for model: %s", m.ID)
			}
		}
	}
	for _, m := range lib.Modes {
		if m.ID == "" || len(m.Variants) == 0 {
			return nil, fmt.Errorf("invalid mode: %s", m.ID)
		}
	}
	return &JSONRepository{lib}, nil
}
func (r *JSONRepository) Metadata() map[string]any {
	count := 0
	for _, m := range r.library.Modes {
		count += len(m.Variants)
	}
	return map[string]any{"source": r.library.Source, "generatedAt": r.library.GeneratedAt, "modelCount": len(r.library.Models), "modeCount": len(r.library.Modes), "variantCount": count}
}
func (r *JSONRepository) Categories() []json.RawMessage { return r.library.Categories }
func (r *JSONRepository) Models(f Filters) []Model {
	result := make([]Model, 0)
	for _, m := range r.library.Models {
		category := strings.Join(m.CategoryPath, "/")
		if f.Category != "" && category != f.Category && !strings.HasPrefix(category, f.Category+"/") {
			continue
		}
		if f.Status != "" && m.Status != f.Status {
			continue
		}
		if f.Kind != "" && m.Kind != f.Kind {
			continue
		}
		searchText := m.Name + " " + m.EnglishName + " " + m.Definition
		if m.Reading != nil {
			searchText += " " + m.Reading.Summary + " " + m.Reading.Title
		}
		if f.Query != "" && !strings.Contains(strings.ToLower(searchText), strings.ToLower(f.Query)) {
			continue
		}
		result = append(result, m)
	}
	return result
}
func (r *JSONRepository) Model(id string) (Model, bool) {
	for _, m := range r.library.Models {
		if m.ID == id {
			return m, true
		}
	}
	return Model{}, false
}
func (r *JSONRepository) Modes() []Mode { return r.library.Modes }
func (r *JSONRepository) Mode(id string) (Mode, bool) {
	for _, m := range r.library.Modes {
		if m.ID == id {
			return m, true
		}
	}
	return Mode{}, false
}
func (r *JSONRepository) Associations() []json.RawMessage { return r.library.Associations }
func respond(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(data)
}
func Router(repo Repository) http.Handler {
	r := chi.NewRouter()
	r.Use(middleware.Recoverer)
	r.Get("/api/meta", func(w http.ResponseWriter, req *http.Request) { respond(w, 200, repo.Metadata()) })
	r.Get("/api/categories", func(w http.ResponseWriter, req *http.Request) { respond(w, 200, repo.Categories()) })
	r.Get("/api/models", func(w http.ResponseWriter, req *http.Request) {
		q := req.URL.Query()
		respond(w, 200, repo.Models(Filters{strings.TrimSpace(q.Get("q")), q.Get("category"), q.Get("status"), q.Get("kind")}))
	})
	r.Get("/api/models/{id}", func(w http.ResponseWriter, req *http.Request) {
		m, ok := repo.Model(chi.URLParam(req, "id"))
		if !ok {
			respond(w, 404, map[string]string{"error": "模型不存在"})
			return
		}
		respond(w, 200, m)
	})
	r.Get("/api/modes", func(w http.ResponseWriter, req *http.Request) { respond(w, 200, repo.Modes()) })
	r.Get("/api/modes/{id}", func(w http.ResponseWriter, req *http.Request) {
		m, ok := repo.Mode(chi.URLParam(req, "id"))
		if !ok {
			respond(w, 404, map[string]string{"error": "曲线模式不存在"})
			return
		}
		respond(w, 200, m)
	})
	r.Get("/api/associations", func(w http.ResponseWriter, req *http.Request) { respond(w, 200, repo.Associations()) })
	r.NotFound(func(w http.ResponseWriter, req *http.Request) {
		respond(w, 404, map[string]string{"error": "资源不存在"})
	})
	return r
}
func main() {
	filename := os.Getenv("LIBRARY_PATH")
	if filename == "" {
		filename = "data/library.json"
	}
	repo, err := LoadRepository(filename)
	if err != nil {
		log.Fatal(err)
	}
	addr := os.Getenv("API_ADDR")
	if addr == "" {
		addr = "127.0.0.1:8080"
	}
	server := &http.Server{Addr: addr, Handler: Router(repo), ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 15 * time.Second, WriteTimeout: 15 * time.Second, IdleTimeout: 60 * time.Second}
	log.Printf("LollaNet read-only API http://%s (%d models)", addr, len(repo.library.Models))
	log.Fatal(server.ListenAndServe())
}
