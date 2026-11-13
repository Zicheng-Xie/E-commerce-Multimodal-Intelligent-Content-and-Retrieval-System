/* Local-only human decisions. No machine labels or predicted scores are supplied. */
(function () {
  "use strict";
  const fields = ["task_id", "reviewer_type", "reviewer", "reviewed_at", "human_grade", "reason"];
  function csvCell(value) {
    return '"' + String(value).replaceAll('"', '""') + '"';
  }
  function completed(row) {
    return row && row.reviewer_type === "human" && row.reviewer.trim() && row.reviewed_at &&
      ["0", "1", "uncertain"].includes(row.human_grade) && row.reason.trim();
  }
  function exportCsv(rows) {
    const ready = rows.filter(completed);
    return "\uFEFF" + fields.join(",") + "\r\n" + ready.map(row =>
      fields.map(field => csvCell(row[field])).join(",")).join("\r\n") + "\r\n";
  }
  function start() {
    const batch = document.body.dataset.batch;
    const packet = document.body.dataset.packet;
    const key = "module2-human-20261105-" + (packet ? packet + "-" : "") + batch;
    const reviewer = document.getElementById("reviewer");
    const human = document.getElementById("human");
    const progress = document.getElementById("progress");
    const cards = [...document.querySelectorAll("article[data-task]")];
    let state = {};
    let storageAvailable = true;
    try { state = JSON.parse(localStorage.getItem(key) || "{}"); }
    catch (_) { storageAvailable = false; }
    reviewer.value = state.reviewer || "";
    state.rows = state.rows || {};
    function save() {
      state.reviewer = reviewer.value.trim();
      try { localStorage.setItem(key, JSON.stringify(state)); }
      catch (_) { storageAvailable = false; }
      const count = cards.filter(card => completed(state.rows[card.dataset.task])).length;
      progress.textContent = "本批已完成 " + count + "/" + cards.length +
        (storageAvailable ? "；已尝试保存到本浏览器，请导出备份。" : "；浏览器存储不可用，请立即导出备份。");
    }
    cards.forEach(card => {
      const id = card.dataset.task;
      const row = state.rows[id];
      const reason = card.querySelector(".reason");
      if (row) {
        reason.value = row.reason || "";
        const radio = [...card.querySelectorAll('input[type="radio"]')]
          .find(item => item.value === row.human_grade);
        if (radio) radio.checked = true;
      }
      card.addEventListener("input", () => {
        if (!human.checked || !reviewer.value.trim()) {
          card.querySelector(".saved").textContent = "请先填写审核者并确认本人独立判断；当前输入尚未记为完成。";
          return;
        }
        const selected = card.querySelector('input[type="radio"]:checked');
        state.rows[id] = {task_id:id, reviewer_type:"human", reviewer:reviewer.value.trim(),
          reviewed_at:new Date().toISOString(), human_grade:selected ? selected.value : "",
          reason:reason.value.trim()};
        card.querySelector(".saved").textContent = completed(state.rows[id]) ? "判断已暂存，请导出备份。" : "请选择判断并填写理由。";
        save();
      });
    });
    function setEnabled() {
      const enabled = human.checked && Boolean(reviewer.value.trim());
      cards.forEach(card => card.querySelectorAll("input").forEach(input => { input.disabled = !enabled; }));
      save();
    }
    reviewer.addEventListener("input", setEnabled);
    human.addEventListener("change", setEnabled);
    document.getElementById("export").addEventListener("click", () => {
      if (!human.checked || !reviewer.value.trim()) {
        alert("请填写审核者，并确认由你本人独立完成判断。"); return;
      }
      const rows = cards.map(card => state.rows[card.dataset.task]).filter(completed);
      if (!rows.length) { alert("尚无判断和理由均完整的答案。"); return; }
      const url = URL.createObjectURL(new Blob([exportCsv(rows)], {type:"text/csv;charset=utf-8"}));
      const link = document.createElement("a");
      link.href = url;
      link.download = (packet ? packet + "_" : "") + "human_batch_" + batch + ".csv";
      document.body.appendChild(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    setEnabled();
  }
  if (typeof module !== "undefined" && module.exports) module.exports = {exportCsv, completed};
  else start();
}());
