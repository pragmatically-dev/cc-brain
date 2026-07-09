# Plan v0.4 — retrieval matemático (spec de ejecución)

> Escrito por el supervisor (Fable). Ejecuta UN builder (Sonnet) fase por fase,
> con TDD estricto, `ruff` limpio y un commit por fase. Después de cada fase:
> STOP y reporte al supervisor. NO avanzar a la fase siguiente sin orden.
>
> Estado actual relevante (verificado en código): la fusión BM25+vector YA es
> RRF (`1/(60+rank)`, `indexer.py` en `search()`); ya existe bonus de trust
> (`0.02*trust`) y decay de recencia fijo (`0.015*exp(-age_days/21)`).
> No re-implementar eso.

## Reglas para todas las fases

- TDD: test primero, verlo fallar, implementar mínimo, verde, refactor.
- Sin dependencias nuevas (numpy ya está; NO scipy/networkx/sklearn).
- Los tests usan los mocks de `tests/conftest.py` (`brain_home`) y el patrón
  `FakeEmbedder`/`embed_env` de `tests/test_audit_v03.py`. Ningún test debe
  inicializar fastembed real ni turbovec real.
- `python -m pytest -q` completo y `ruff check cc_brain tests` limpios antes
  de cada commit.
- Compatibilidad: DBs existentes sin migración manual — todo cambio de esquema
  vía `ALTER TABLE ... ` con `try/except OperationalError` en `store.py`
  (patrón ya usado en el proyecto) o tabla nueva `CREATE TABLE IF NOT EXISTS`.
- Mensajes de commit: `feat(retrieval): ...` en inglés, cuerpo explicando el
  modelo matemático en 2-4 líneas.

---

## FASE 1 — MMR (diversidad) + half-life por tipo de fuente

### 1a. MMR en `search()`

Problema: el top-k puede ser 6 variantes del mismo contenido; la única defensa
es el cap `per_path>=2`, que no ve similitud semántica.

Implementación (en `indexer.py`):

- Helper `def _mmr(candidates, vecs, k, lam=0.75)`: entrada = lista ordenada
  `(adjusted, cid, row)` + dict `cid -> np.ndarray` (embeddings normalizados);
  greedy clásico: elegir el próximo `argmax lam*rel_norm(c) - (1-lam)*max_cos(c, seleccionados)`.
  `rel_norm` = score min-max normalizado dentro del pool de candidatos.
- En `search()`, tras `candidates.sort(...)`: si `not lex` y hay embeddings
  disponibles, cargar los vecs de los top `min(len(candidates), 4*k)` cids
  con un solo `SELECT chunk, vec FROM embeddings WHERE chunk IN (...)`,
  `np.frombuffer(..., dtype=np.float32)`, normalizar L2 (dividir por norma,
  proteger norma 0). Aplicar `_mmr` sobre ese pool y DESPUÉS el cap
  `per_path` existente sobre el orden que devuelve MMR.
- Si un cid del pool no tiene embedding (carrera con re-embed) → tratarlo como
  ortogonal (max_cos=0), no excluirlo.
- `lex=True` mantiene el camino actual sin MMR (no hay vecs en juego).

Tests (`tests/test_math_v04.py`):
- 4 chunks: 3 con embedding idéntico `[1,0,0,0]`, 1 con `[0,1,0,0]`, scores
  fusionados similares; con k=3 el resultado debe incluir el chunk ortogonal
  (con el orden por score puro no entraría). Monkeypatchear `_vec_search` para
  devolver los 4 y `_bm25` vacío, embeddings insertados a mano en la tabla.
- Caso degenerado: pool de 1 → devuelve ese único.
- `lex=True` → orden idéntico al actual (sin tocar embeddings).

### 1b. Half-life de recencia por tipo de fuente

Problema: `exp(-age_days/21)` fijo trata igual una nota curada (casi atemporal)
que un transcript de sesión (decae rápido).

Implementación:

- En `indexer.py`: `RECENCY_HALF_LIFE_DAYS = {"session": 14.0, "web": 45.0,
  "code": 90.0, "memory": 365.0}` con default `30.0`. La clave es
  `SourceSpec.kind` (los kinds reales: verificar en `config.py` /
  `contracts.py` y usar los que existan — ajustar el dict a los kinds reales,
  no inventar).
- En `search()`: construir una vez `kind_by_source = {s.name: s.kind for s in
  load_sources(p)}` y reemplazar el término fijo por
  `0.015 * exp(-ln(2) * age_days / half_life)`.
- `doctor()`/`stats()` NO cambian.

Tests: dos chunks con el mismo score fusionado y misma edad (p.ej. 30 días),
uno de source kind `session` y otro `memory` → el de `memory` rankea arriba.

**Commit y STOP. Reportar: diff resumido, salida de pytest, decisiones tomadas
(kinds reales usados).**

---

## FASE 2 — Feedback implícito bayesiano (trust que aprende)

Idea: cada `get(ids)` posterior a un `search()` es señal de relevancia gratis.

Implementación:

- `store.py::ensure_schema` (o donde se crea el esquema): columna nueva
  `chunks.uses INTEGER NOT NULL DEFAULT 0` vía ALTER TABLE tolerante.
- `indexer.get()`: `UPDATE chunks SET uses = uses + 1 WHERE id IN (...)`.
- Ranking en `search()`: bonus `min(0.01 * math.log1p(uses), 0.04)` sumado a
  `adjusted`. Log para saturar (no plutocracia de chunks viejos), cap duro
  para que nunca domine sobre relevancia.
- `stats()`: agregar `top_used` (5 chunks más usados: id, path, uses).

Tests: (1) `get()` incrementa `uses`; (2) dos chunks con mismo score, uno con
uses=20 → rankea arriba; (3) el bonus está capeado (uses=10^6 no supera el cap).

**Commit y STOP. Reportar.**

---

## FASE 3 — Grafo de memoria + PageRank personalizado

Idea: recuperar lo estructuralmente central, no solo lo textualmente similar.

Implementación:

- Tabla `CREATE TABLE IF NOT EXISTS edges(a INTEGER, b INTEGER, w REAL,
  PRIMARY KEY(a,b))`, poblada al final de `index()` (solo para archivos
  changed, borrando antes sus edges) con estas aristas no dirigidas
  (guardar con a<b):
  - mismo `path`: w=1.0 entre chunks consecutivos por `loc` (cadena, no clique
    completo — un archivo de 200 chunks no debe generar 20k aristas);
  - wikilinks `[[nombre]]` en el texto de notas → arista w=2.0 al chunk título
    de la nota destino si existe (matchear por `path` que termine en
    `<slug>.md` — reusar `slug()` de config);
  - misma sesión/mtime cercano: chunks del mismo source con `|Δmtime| < 3600`
    y path distinto → w=0.5, SOLO entre archivos consecutivos por mtime
    (de nuevo cadena, no clique).
- `search()`: tras fusion RRF, personalized PageRank restringido:
  - nodos = candidatos fusionados + sus vecinos directos (1 salto, un SELECT
    sobre `edges`), tope 512 nodos;
  - teleport vector = scores fusionados normalizados (0 para vecinos puros);
  - 8 iteraciones de `r = 0.85 * (W_norm @ r) + 0.15 * teleport` con numpy
    denso (512x512 max = trivial);
  - blend: `adjusted += 0.05 * ppr_norm` (min-max sobre el subgrafo).
  - Si no hay edges (tabla vacía) → skip silencioso, cero costo.
- CLI/MCP sin cambios de interfaz.

Tests: grafo sintético insertado a mano (A-B fuerte, C aislado, scores
fusionados iguales) → B (vecino del seed A) supera a C. Test de que sin edges
el ranking es idéntico al de Fase 1/2 (no-regresión).

**Commit y STOP. Reportar.**

---

## FASE 4 — Near-duplicados + empaquetado submodular de project_state

### 4a. `cc-brain dedupe` (reporte, NUNCA borra)

- Nuevo comando CLI `dedupe [--threshold 0.95] [--limit 50]`: recorre
  embeddings por lotes de 512, normaliza, producto matricial contra el corpus
  por bloques (o usa `_vec_search` por chunk si es más simple), reporta pares
  cross-path con cos > threshold: `[cid_a] path_a  ~ [cid_b] path_b  (0.97)`.
- `doctor()`: warning si hay >20 pares duplicados ("run cc-brain dedupe").
  Para no encarecer doctor: contar solo sobre una muestra de 256 chunks random.

### 4b. Submodular packing en `project_snapshot`

- Donde el snapshot junta chunks relacionados: reemplazar top-k plano por
  greedy facility-location con presupuesto: iterar candidatos, agregar el que
  maximiza `rel(c) - max_cos(c, seleccionados)` (reusar `_mmr` con lam=0.7)
  hasta agotar un presupuesto de ~4000 caracteres (aprox tokens*4).
- Mantener SIEMPRE el atom de sesión más nuevo y los commits recientes
  (determinismo actual intacto); el packing aplica solo a la parte "related
  chunks".

Tests: dedupe encuentra el par plantado y no reporta self-pairs ni same-path;
packing respeta presupuesto y prefiere diverso sobre duplicado.

**Commit y STOP. Reportar.**

---

## FASE 5 (documentación y cierre)

- `llm.md`: sección corta "Ranking model" explicando: RRF → trust + half-life
  por kind + uses aprendidos + PageRank sobre el grafo de memoria → MMR.
  Una fórmula por término, 10 líneas máximo.
- `README.md`: bullet de features actualizado.
- `pyproject.toml`: bump versión a 0.4.0.
- Commit final `release: v0.4.0 math retrieval upgrade`.

## Explicitamente FUERA de alcance (no hacer)

- OPQ/rotaciones del índice turbovec, Matryoshka (requiere tocar turbovec).
- Chunking semántico por embeddings (cambia todos los cid — otra release).
- Cross-encoder reranking (dependencia nueva).
- Cualquier auto-borrado de datos.
