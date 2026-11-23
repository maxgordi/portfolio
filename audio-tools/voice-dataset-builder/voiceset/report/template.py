TEMPLATE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Voice Dataset Builder — отчёт</title>
<style>
:root{--bg:#fff;--fg:#1f2937;--mut:#6b7280;--line:#e5e7eb;--card:#f9fafb;--acc:#2563eb;--bad:#dc2626;--warn:#b45309}
@media (prefers-color-scheme:dark){:root{--bg:#111827;--fg:#e5e7eb;--mut:#9ca3af;--line:#374151;--card:#1f2937;--acc:#60a5fa;--bad:#f87171;--warn:#fbbf24}}
body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 10px}
.mut{color:var(--mut)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px}.card b{display:block;font-size:22px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}
th{color:var(--mut);font-weight:600;position:sticky;top:0;background:var(--bg)}audio{height:30px;width:220px}
.warn{background:var(--card);border-left:4px solid var(--warn);padding:8px 12px;margin:6px 0;border-radius:6px}
.ok{border-left-color:var(--acc)}.tag{display:inline-block;padding:1px 7px;border-radius:99px;border:1px solid var(--line);font-size:12px}
.bad{color:var(--bad)}img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}
.wrap{overflow-x:auto}details summary{cursor:pointer;margin:8px 0}
</style></head><body><main>
<h1>Voice Dataset Builder</h1>
<div class="mut">{{ stamp }} · seed {{ seed }} · сходство: {{ speaker }}</div>

<div class="cards">
<div class="card"><span class="mut">Вход</span><b>{{ input_min }} мин</b></div>
<div class="card"><span class="mut">Фраз всего</span><b>{{ n_all }}</b></div>
<div class="card"><span class="mut">Чистых</span><b>{{ n_clean }}</b></div>
<div class="card"><span class="mut">Отброшено</span><b>{{ n_rej }}</b></div>
<div class="card"><span class="mut">В наборе</span><b>{{ n_sel }} фраз</b></div>
<div class="card"><span class="mut">Длина набора</span><b>{{ length }} с</b></div>
<div class="card"><span class="mut">Для RVC</span><b>{{ rvc_min }} мин</b></div>
</div>

<h2>Предупреждения</h2>
{% for w in warnings %}<div class="warn">{{ w }}</div>{% else %}<div class="warn ok">Замечаний нет.</div>{% endfor %}
{% if diff %}<div class="warn ok">Отличия от прошлой сборки: добавлено {{ diff.added or '—' }}, убрано {{ diff.removed or '—' }}.</div>{% endif %}

<h2>Карта диапазона</h2>
<p class="mut">Серые столбики — сколько секунд чистого пения на ноте в записи, синие — сколько попало в набор. Красная полоса — нота диапазона, не попавшая в набор.</p>
<img alt="карта диапазона" src="data:image/png;base64,{{ img_range }}">

<h2>Доли подач</h2>
<img alt="доли подач" src="data:image/png;base64,{{ img_deliv }}">

<h2>Выбранные фразы (в порядке набора)</h2>
<div class="wrap"><table><tr><th>№</th><th>id</th><th>Песня</th><th>Время в исходнике</th><th>Диапазон</th><th>Подача</th><th>Качество</th><th>Сходство</th><th>Плеер</th></tr>
{% for r in selected %}<tr><td>{{ r.order }}</td><td>{{ r.id }}</td><td>{{ r.song }}</td><td>{{ r.t }}</td><td>{{ r.range }}</td>
<td><span class="tag">{{ r.delivery }}</span> {{ r.conf }}{% if r.unsure %} <span class="mut">не уверен</span>{% endif %}</td><td>{{ r.quality }}</td><td>{{ r.sim }}</td>
<td><audio controls preload="none" src="../{{ r.path }}"></audio></td></tr>{% endfor %}</table></div>

<h2>Выброшенные фразы</h2>
<div class="wrap"><table><tr><th>id</th><th>Песня</th><th>Время</th><th>Причина</th><th>Плеер</th></tr>
{% for r in rejected %}<tr><td>{{ r.id }}</td><td>{{ r.song }}</td><td>{{ r.t }}</td><td class="bad">{{ r.why }}</td>
<td><audio controls preload="none" src="../{{ r.path }}"></audio></td></tr>{% else %}<tr><td colspan="5" class="mut">Ничего не выброшено.</td></tr>{% endfor %}</table></div>

<details><summary>Чистые фразы, не попавшие в набор ({{ spare|length }})</summary>
<div class="wrap"><table><tr><th>id</th><th>Песня</th><th>Время</th><th>Диапазон</th><th>Подача</th><th>Почему не взята</th><th>Плеер</th></tr>
{% for r in spare %}<tr><td>{{ r.id }}</td><td>{{ r.song }}</td><td>{{ r.t }}</td><td>{{ r.range }}</td><td>{{ r.delivery }}</td><td class="mut">{{ r.why }}</td>
<td><audio controls preload="none" src="../{{ r.path }}"></audio></td></tr>{% endfor %}</table></div></details>

<h2>Как править</h2>
<p>Слушайте выбранные фразы, запишите номера (<b>id</b>), которые не нравятся, и запустите:<br>
<code>voiceset build-suno {{ root }} --exclude 12,37 --include 5</code><br>
Или отредактируйте <code>review.yaml</code> (keep — взять, ban — не брать). Пересборка детерминирована; отчёт покажет разницу с прошлой.</p>
</main></body></html>
"""
