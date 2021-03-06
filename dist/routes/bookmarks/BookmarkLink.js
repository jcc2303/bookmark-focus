/* src/routes/bookmarks/BookmarkLink.svelte generated by Svelte v3.31.1 */
import {
	SvelteComponent,
	append,
	attr,
	create_component,
	destroy_component,
	detach,
	element,
	init,
	insert,
	listen,
	mount_component,
	run_all,
	safe_not_equal,
	set_data,
	space,
	text,
	transition_in,
	transition_out
} from "../../../web_modules/svelte/internal.js";

import { createEventDispatcher } from "../../../web_modules/svelte.js";
import { ExternalLinkIcon, CpuIcon, Trash2Icon } from "../../../web_modules/svelte-feather-icons.js";
import chromeApi from "../../background/chrome-api.js";

function create_if_block(ctx) {
	let p;
	let t_value = /*path*/ ctx[4].replace(/*relative*/ ctx[0], "") + "";
	let t;

	return {
		c() {
			p = element("p");
			t = text(t_value);
			attr(p, "class", "px-1 text-xs  text-blue-600");
		},
		m(target, anchor) {
			insert(target, p, anchor);
			append(p, t);
		},
		p(ctx, dirty) {
			if (dirty & /*path, relative*/ 17 && t_value !== (t_value = /*path*/ ctx[4].replace(/*relative*/ ctx[0], "") + "")) set_data(t, t_value);
		},
		d(detaching) {
			if (detaching) detach(p);
		}
	};
}

function create_fragment(ctx) {
	let li;
	let div1;
	let img;
	let img_src_value;
	let t0;
	let t1;
	let span0;
	let t2;
	let t3;
	let span5;
	let a;
	let span1;
	let t4;
	let t5;
	let span2;
	let externallinkicon;
	let t6;
	let div0;
	let span3;
	let button0;
	let cpuicon;
	let t7;
	let span4;
	let button1;
	let trash2icon;
	let current;
	let mounted;
	let dispose;
	let if_block = /*relative*/ ctx[0] && /*path*/ ctx[4] && create_if_block(ctx);
	externallinkicon = new ExternalLinkIcon({ props: { size: "1x" } });
	cpuicon = new CpuIcon({ props: { size: "1x" } });
	trash2icon = new Trash2Icon({ props: { size: "1x" } });

	return {
		c() {
			li = element("li");
			div1 = element("div");
			img = element("img");
			t0 = space();
			if (if_block) if_block.c();
			t1 = space();
			span0 = element("span");
			t2 = text(/*title*/ ctx[2]);
			t3 = space();
			span5 = element("span");
			a = element("a");
			span1 = element("span");
			t4 = text(/*url*/ ctx[1]);
			t5 = space();
			span2 = element("span");
			create_component(externallinkicon.$$.fragment);
			t6 = space();
			div0 = element("div");
			span3 = element("span");
			button0 = element("button");
			create_component(cpuicon.$$.fragment);
			t7 = space();
			span4 = element("span");
			button1 = element("button");
			create_component(trash2icon.$$.fragment);
			if (img.src !== (img_src_value = /*favicon*/ ctx[3])) attr(img, "src", img_src_value);
			attr(img, "class", "px-2");
			attr(img, "alt", "");
			attr(span0, "class", "flex truncate  opacity-100 group-hover:opacity-0");
			attr(span1, "class", "truncate");
			attr(a, "href", /*url*/ ctx[1]);
			attr(a, "target", "_blank");
			attr(a, "class", "opacity-0 group-hover:opacity-100 bg-gray-300 text-blue-500 flex truncate items-center content-center");
			attr(button0, "class", "px-1");
			attr(span3, "class", "flex right-0 opacity-0 group-hover:opacity-100");
			attr(button1, "class", "px-1");
			attr(span4, "class", "flex right-0 opacity-0 group-hover:opacity-100");
			attr(div0, "class", "float-right flex");
			attr(span5, "class", "cursor-pointer absolute group-hover:bg-gray-300 flex w-full justify-between");
			attr(div1, "class", "flex group hover:bg-gray-200 items-center relative truncate");
			attr(li, "class", "bg-gray-100 hover:border");
		},
		m(target, anchor) {
			insert(target, li, anchor);
			append(li, div1);
			append(div1, img);
			append(div1, t0);
			if (if_block) if_block.m(div1, null);
			append(div1, t1);
			append(div1, span0);
			append(span0, t2);
			append(div1, t3);
			append(div1, span5);
			append(span5, a);
			append(a, span1);
			append(span1, t4);
			append(a, t5);
			append(a, span2);
			mount_component(externallinkicon, span2, null);
			append(span5, t6);
			append(span5, div0);
			append(div0, span3);
			append(span3, button0);
			mount_component(cpuicon, button0, null);
			append(div0, t7);
			append(div0, span4);
			append(span4, button1);
			mount_component(trash2icon, button1, null);
			current = true;

			if (!mounted) {
				dispose = [
					listen(button0, "click", /*doAnalisys*/ ctx[5]),
					listen(button1, "click", /*remove*/ ctx[6])
				];

				mounted = true;
			}
		},
		p(ctx, [dirty]) {
			if (!current || dirty & /*favicon*/ 8 && img.src !== (img_src_value = /*favicon*/ ctx[3])) {
				attr(img, "src", img_src_value);
			}

			if (/*relative*/ ctx[0] && /*path*/ ctx[4]) {
				if (if_block) {
					if_block.p(ctx, dirty);
				} else {
					if_block = create_if_block(ctx);
					if_block.c();
					if_block.m(div1, t1);
				}
			} else if (if_block) {
				if_block.d(1);
				if_block = null;
			}

			if (!current || dirty & /*title*/ 4) set_data(t2, /*title*/ ctx[2]);
			if (!current || dirty & /*url*/ 2) set_data(t4, /*url*/ ctx[1]);

			if (!current || dirty & /*url*/ 2) {
				attr(a, "href", /*url*/ ctx[1]);
			}
		},
		i(local) {
			if (current) return;
			transition_in(externallinkicon.$$.fragment, local);
			transition_in(cpuicon.$$.fragment, local);
			transition_in(trash2icon.$$.fragment, local);
			current = true;
		},
		o(local) {
			transition_out(externallinkicon.$$.fragment, local);
			transition_out(cpuicon.$$.fragment, local);
			transition_out(trash2icon.$$.fragment, local);
			current = false;
		},
		d(detaching) {
			if (detaching) detach(li);
			if (if_block) if_block.d();
			destroy_component(externallinkicon);
			destroy_component(cpuicon);
			destroy_component(trash2icon);
			mounted = false;
			run_all(dispose);
		}
	};
}

function instance($$self, $$props, $$invalidate) {
	let dispatch = createEventDispatcher();
	let { link } = $$props;
	let { relative = "" } = $$props;
	let id, url, title, parentId;
	let favicon, path;

	const getFavicon = async url => {
		// console.log(url);
		$$invalidate(3, favicon = link.icon || await chromeApi.extras.fetchFavicon(url));
	};

	const doAnalisys = () => {
		console.log("doAnalisys", link);
		window.dispatchEvent(new CustomEvent("worker.analisys", { detail: [link] }));
	};

	const remove = () => {
		dispatch("remove", link);
	};

	$$self.$$set = $$props => {
		if ("link" in $$props) $$invalidate(7, link = $$props.link);
		if ("relative" in $$props) $$invalidate(0, relative = $$props.relative);
	};

	$$self.$$.update = () => {
		if ($$self.$$.dirty & /*relative*/ 1) {
			$: $$invalidate(0, relative = relative === "/" ? /\/\/.*?(?=\/)/g : relative);
		}

		if ($$self.$$.dirty & /*link, url*/ 130) {
			$: if (link) {
				id = link.id;
				$$invalidate(1, url = link.url);
				$$invalidate(2, title = link.title);
				parentId = link.parentId;
				$$invalidate(4, path = link.parent?.path);

				// chromeApi.extras.getPath(parentId).then(p => path= p)
				getFavicon(url);
			}
		}
	};

	return [relative, url, title, favicon, path, doAnalisys, remove, link];
}

class BookmarkLink extends SvelteComponent {
	constructor(options) {
		super();
		init(this, options, instance, create_fragment, safe_not_equal, { link: 7, relative: 0 });
	}
}

export default BookmarkLink;