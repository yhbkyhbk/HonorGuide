export default function(component) {
    const state = window.__honorguideClock ||= {};
    state.send = value => component.setTriggerValue("measurement", value);
    const data = component.data || {};
    if (state.runId !== data.run_id) {
        state.runId = data.run_id;
        // 新一轮脚本没有接收问题时，解除因切页、停止等遗留的提交锁。
        if (!data.request_id) state.busy = false;
    }
    state.syncSubmit?.();
    if (state.installed) return;
    state.installed = true;
    const answers = () => [...document.querySelectorAll('[class*="st-key-hg_answer_"]')];
    const answerId = node => [...node.classList]
        .find(name => /^st-key-hg_answer_[a-f0-9]{32}$/.test(name))?.slice(17);
    state.syncSubmit = () => {
        const input = document.querySelector('[data-testid="stChatInputTextArea"]');
        const button = document.querySelector('[data-testid="stChatInputSubmitButton"]');
        if (!input || !button) return;
        const disabled = Boolean(state.busy || input.disabled || !input.value.trim());
        if (button.disabled !== disabled) button.disabled = disabled;
    };
    const submit = event => {
        const input = document.querySelector('[data-testid="stChatInputTextArea"]');
        if (!input || input.disabled) return;
        const keyboard = event.type === "keydown" && event.target === input &&
            event.key === "Enter" && !event.shiftKey && !event.isComposing;
        const button = event.type === "click" &&
            event.target.closest('[data-testid="stChatInputSubmitButton"]');
        if (!keyboard && !button) return;
        if (state.busy || (keyboard && event.repeat)) {
            event.preventDefault();
            event.stopImmediatePropagation();
            state.syncSubmit();
            return;
        }
        if (!input.value.trim()) return;
        state.busy = true;
        state.pending = {start: performance.now(), existing: new Set(answers().map(answerId))};
        // 先让本次原生提交完成，再锁定按钮，避免阻止第一次发送。
        queueMicrotask(state.syncSubmit);
    };
    const check = () => {
        state.syncSubmit();
        const p = state.pending;
        if (!p || p.sent) return;
        if (!p.id) {
            const node = answers().find(node => !p.existing.has(answerId(node)));
            if (node) p.id = answerId(node);
        }
        if (!p.id) return;
        const done = document.querySelector('.st-key-hg_done_' + p.id);
        if (done) {
            state.busy = false;
            state.syncSubmit();
        }
        const node = document.querySelector('.st-key-hg_answer_' + p.id);
        const text = node?.querySelector('[data-testid="stMarkdownContainer"]')?.textContent.trim();
        if (text && p.first === undefined && !p.scheduled && document.visibilityState === "visible") {
            p.scheduled = true;
            p.detected = performance.now() - p.start;
            // 双 rAF 近似首次文本绘制；不声称获得物理屏幕的精确刷新时刻。
            requestAnimationFrame(() => requestAnimationFrame(() => {
                if (state.pending !== p) return;
                p.scheduled = false;
                if (!node.isConnected || document.visibilityState !== "visible") {
                    check();
                    return;
                }
                p.first = performance.now() - p.start;
                check();
            }));
        }
        if (p.first !== undefined && done) {
            p.sent = true;
            state.send({request_id: p.id, browser_first_render_ms: p.first,
                browser_text_detected_ms: p.detected, browser_render_wait_ms: p.first - p.detected});
        }
    };
    document.addEventListener("keydown", submit, true);
    document.addEventListener("click", submit, true);
    document.addEventListener("input", state.syncSubmit, true);
    document.addEventListener("visibilitychange", check);
    state.observer = new MutationObserver(check);
    state.observer.observe(document.body, {subtree: true, childList: true, characterData: true,
        attributes: true, attributeFilter: ["disabled"]});
    check();
}
