import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";
import { $el } from "../../../scripts/ui.js";

app.registerExtension({
	name: "preview.Combo++",
	init() {
		$el("style", {
			textContent: `
				.litemenu-entry {position: relative;}
				.litemenu-entry:hover .preview-combo-image {display: block;}
				.preview-combo-image {
					display: none;
					position: absolute;
					left: 120px;
					top: -120px;
					width: 384px;
					height: 384px;
					background-size: contain;
					background-position: top right;
					background-repeat: no-repeat;
					filter: brightness(65%);
				}
			`,
			parent: document.body
		});

		const submenuSetting = app.ui.settings.addSetting({
			id: "preview.Combo++.Submenu",
			name: "Enable submenu in custom nodes",
			defaultValue: true,
			type: "boolean"
		});

		const getOrSet = (target, name, create) => name in target ? target[name] : (target[name] = create());
		const symbol = getOrSet(window, "__preview__", () => Symbol("__preview__"));
		const store = getOrSet(window, symbol, () => ({}));
		const contextMenuHook = getOrSet(store, "contextMenuHook", () => ({}));
		for (const e of ["ctor", "preAddItem", "addItem"]) {
			if (!contextMenuHook[e]) contextMenuHook[e] = [];
		}

		const isCustomItem = (value) => value && typeof value === "object" && "preview" in value && value.content;
		const splitBy = (navigator.platform || navigator.userAgent).includes("Win") ? /\/|\\/ : /\//;

		contextMenuHook["ctor"].push(function (values, options) {
			if (options.parentMenu?.options?.className === "dark") options.className = "dark";
		});

		contextMenuHook["addItem"].push(function (el, menu, [name, value, options]) {
			if (el && isCustomItem(value) && value?.preview && !value.submenu) {
				$el("div.preview-combo-image", {
					parent: el,
					style: {backgroundImage: `url(/preview/${encodeURIComponent(value.preview)})`}
				});
			}
		});

		function buildMenu(widget, values) {
			const lookup = {"": {options: []}};
			for (const value of values) {
				const split = value.content.split(splitBy);
				let path = "";
				for (let i = 0; i < split.length; i++) {
					const s = split[i];
					const last = i === split.length - 1;
					if (last) {
						lookup[path].options.push({
							...value,
							title: s,
							callback: () => {
								widget.value = value;
								widget.callback(value);
								app.graph.setDirtyCanvas(true);
							}
						});
					} else {
						const prevPath = path;
						path += s + splitBy;
						if (!lookup[path]) {
							const sub = {
								title: s,
								submenu: {options: [], title: s}
							};
							lookup[path] = sub.submenu;
							lookup[prevPath].options.push(sub);
						}
					}
				}
			}
			return lookup[""].options;
		}

		const combo = ComfyWidgets["COMBO"];
		ComfyWidgets["COMBO"] = function (node, inputName, inputData) {
			const type = inputData[0];
			const res = combo.apply(this, arguments);
			if (isCustomItem(type[0])) {
				let value = res.widget.value;
				let values = res.widget.options.values;
				let menu = null;

				Object.defineProperty(res.widget.options, "values", {
					get() {
						let v = values;
						if (submenuSetting.value) {
							if (!menu) menu = buildMenu(res.widget, values);
							v = menu;
						}
						const valuesIncludes = v.includes;
						v.includes = function (searchElement) {
							const includesFromMenuItems = function (items) {
								for (const item of items) {
									if (includesFromMenuItem(item)) return true;
								}
								return false;
							};
							const includesFromMenuItem = function (item) {
								return item.submenu ? includesFromMenuItems(item.submenu.options) : item.content === searchElement.content;
							};
							return valuesIncludes.apply(this, arguments) || includesFromMenuItems(this);
						};
						return v;
					},
					set(v) {
						values = v;
						menu = null;
					}
				});

				Object.defineProperty(res.widget, "value", {
					get() {
						if (res.widget) {
							const stack = new Error().stack;
							if (stack.includes("drawNodeWidgets") || stack.includes("saveImageExtraOutput")) {
								return (value || type[0]).content;
							}
						}
						return value;
					},
					set(v) {
						if (v?.submenu) return;
						value = v;
					}
				});
			}
			return res;
		};
	}
});
