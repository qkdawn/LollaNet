export type ModelStatus = 'active' | 'candidate' | 'contested' | 'deprecated';
export type ModelKind = 'world' | 'reasoning' | 'action';
export interface Section { title: string; markdown: string }
export interface Source { label: string; url: string }
export interface Relation { type: string; target: string; rationale: string }
export interface ReadingSource extends Source { id: string; locator: string; note: string }
export interface ModelReading {
  modelId: string; modelVersion: number; revision: number; title: string; summary: string;
  body: string; appendix: string; sources: ReadingSource[]; sourcePath: string;
}
export interface Model {
  id: string; name: string; englishName: string; version: number;
  status: ModelStatus; kind: ModelKind; formalism: boolean;
  categoryPath: string[]; definition: string; sections: Section[];
  sources: Source[]; relations: Relation[]; sourcePath: string;
  reading?: ModelReading;
}
export interface Category { id: string; name: string; count: number; children: Category[] }
export interface Parameter { key: string; label: string; min: number; max: number; step: number; value: number }
export type Formula = 'constant' | 'linear' | 'exponential' | 'goal' | 'logistic' | 'gompertz' | 'peak' | 'overshoot' | 'oscillation';
export interface Variant {
  id: string; name: string; formula: Formula; equation: string;
  direction: number; parameters: Parameter[]; conditions: string;
}
export interface BehaviorMode {
  id: string; name: string; englishName: string; description: string;
  color: string; variants: Variant[];
  sources: (Source & { status: 'verified' | 'pending' })[];
}
export interface Association { modelRef: string; variantId: string; rationale: string }
export interface Library {
  schemaVersion: 1; generatedAt: string; source: string;
  models: Model[]; categories: Category[]; modes: BehaviorMode[]; associations: Association[];
}
export interface Metadata { generatedAt: string; source: string; modelCount: number; modeCount: number; variantCount: number }
