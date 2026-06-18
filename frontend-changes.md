# Frontend Changes: Dark/Light Theme Toggle

## Summary
Added a theme toggle button (sun/moon icon) fixed to the top-right of the page that switches the app between the existing dark theme and a new light theme, with the preference persisted in `localStorage`.

## Files Changed

### `frontend/index.html`
- Added a `<button id="themeToggle" class="theme-toggle">` right after the (hidden) `<header>`, containing two inline SVG icons (`theme-icon-sun`, `theme-icon-moon`).
- Button includes `aria-label`, `aria-pressed`, and `title` attributes for accessibility; labels are updated dynamically by JS to reflect the action that will occur on click.

### `frontend/style.css`
- Added a `:root[data-theme="light"]` block that overrides the existing CSS variables (`--background`, `--surface`, `--text-primary`, etc.) with light-theme values, so the rest of the stylesheet (which already references these variables) repaints automatically.
- Added `transition` rules on `body`, `.main-content`, `.chat-main`, `.chat-container`, `.chat-messages`, and `.chat-input-container` for a smooth color/background fade when switching themes.
- Added `.theme-toggle` styles: fixed position top-right, circular button matching the existing surface/border/shadow aesthetic, hover/focus-visible/active states consistent with other buttons in the app.
- Added `.theme-icon-sun` / `.theme-icon-moon` cross-fade + rotate/scale transition so the icon swap animates smoothly instead of popping.
- Added a small media-query tweak to shrink the button on mobile widths.

### `frontend/script.js`
- Cached the new `themeToggle` element on `DOMContentLoaded`.
- Added `initTheme()`: reads `localStorage.getItem('theme')`, falling back to the OS `prefers-color-scheme` setting, and applies it on load.
- Added `toggleTheme()`: flips between `'light'` and `'dark'`.
- Added `applyTheme(theme)`: sets/removes `data-theme="light"` on `<html>`, persists the choice to `localStorage`, and updates `aria-pressed` / `aria-label` on the button.
- Wired up `click` and `keydown` (Enter/Space) listeners on the toggle button so it's fully keyboard-navigable.

## Behavior
- Defaults to the existing dark theme (no `data-theme` attribute) unless the user previously chose light mode, or their OS prefers light mode and no choice was saved yet.
- Choice persists across reloads via `localStorage`.
- Toggle is reachable via Tab and operable via Enter/Space, with screen-reader-friendly labeling that updates based on current state.

## Verification
Reviewed the HTML/CSS/JS changes manually for correctness. Could not perform a live browser check because Playwright's Chromium binary is not installed in this environment (`npx playwright install chrome` would be required) — this should be verified visually before merging.
