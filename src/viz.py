from pprint import pformat
import textwrap
from playwright.async_api import async_playwright
import asyncio
import os
import json
import statistics
import shutil

class TextCobwebDrawer:

	def __init__(self, attributes, values):

		self.id_to_attr = attributes
		self.attr_to_id = dict([(w, i) for i, w in enumerate(attributes)])

		self.id_to_value = values
		self.value_to_id = dict([(w, i) for i, w in enumerate(values)])

	def _visualize_node(self, node):
		"""
		Helper method to visualize a table for a discrete Cobweb node!

		Given a cobweb_discrete.CobwebNode, this visualizes a table for a given
		cobweb node in terms of the attribute IDs and corpus IDs

		We're just going to find some nice way to return a print of the indented
		list for right now! Better visualizations to come shortly
		"""

		str_dict = {}

		for a, v in node.av_count.items():
				str_dict[self.id_to_attr[a]] = dict([(self.id_to_value[v_id], cnt) for v_id, cnt in v.items()])

		return f"- NODE_{node.concept_hash()}\n" + pformat(str_dict, indent=4)

	def visualize_tree(self, root):
		"""
		Visualizes a discrete Cobweb tree given the root node of the tree -
		recursive retrieval and printing!

		We'll probably instantiate a DFS for this and then use textwrap!
		"""

		def print_dfs(node, depth=0):

				print(textwrap.indent(self._visualize_node(node), prefix="    " * depth))

				for child in node.children:
						print_dfs(child, depth + 1)

		print_dfs(root)

class HTMLCobwebDrawer:
	def __init__(self, attributes, id_to_value, value_to_id,
				 attr_value_fn=None, attr_name_overrides=None):
		"""
		Parameters
		----------
		attributes : list[str]
			Human-readable header names for each attribute index.
		id_to_value : list[str]
			Default value-name lookup (index → display string).
		value_to_id : dict[str, int]
			Reverse mapping of id_to_value.
		attr_value_fn : dict[int, callable] | None
			Optional per-attribute value-name overrides.  Keys are attribute
			indices (matching the attribute order in *attributes*); values are
			callables ``fn(val_id) -> str`` that return the display string for
			a given value id.  Attributes not present in this dict fall back to
			the global ``id_to_value`` list.
		attr_name_overrides : dict[int, str] | None
			Optional mapping from attribute index to display name.  Useful for
			hidden (negative-index) attributes that are not in the positional
			*attributes* list but should still show a meaningful header in the
			visualisation.
		"""
		self.id_to_attr = attributes
		self.attr_to_id = {w: i for i, w in enumerate(attributes)}
		self.id_to_value = id_to_value
		self.value_to_id = value_to_id
		self.attr_value_fn = attr_value_fn or {}
		self.attr_name_overrides = attr_name_overrides or {}

	def _safe_lookup(self, id_to_list, idx):
		return id_to_list[idx] if (idx is not None and 0 <= idx < len(id_to_list)) else "None"

	def _node_to_dict(self, node, max_depth=None, _depth=0):
		"""
		Convert a CobwebNode into a JSON dict for D3 rendering.

		Each node produces a flat list of ``{attr, val, count}`` rows
		(one row per value per attribute) for a single three-column table.
		"""
		title = f"CONCEPT-{node.concept_hash()}"

		rows = []
		for attr_id, val_counts in sorted(node.av_count.items()):
			attr_name = self.attr_name_overrides.get(
				attr_id, self._safe_lookup(self.id_to_attr, attr_id)
			)

			# Sort values by descending count, then take top 7
			top_vals = sorted(val_counts.items(), key=lambda x: x[1], reverse=True)[:7]

			first = True
			for val_id, count in top_vals:
				if attr_id in self.attr_value_fn:
					val_name = self.attr_value_fn[attr_id](val_id)
				else:
					val_name = self._safe_lookup(self.id_to_value, val_id)
				rows.append({
					"attr": attr_name if first else "",
					"val": val_name,
					"count": count,
				})
				first = False

			if len(val_counts) > 7:
				rows.append({"attr": "", "val": "...", "count": "..."})

		# Stop recursion if max_depth is reached
		if max_depth is not None and _depth >= max_depth:
			children = []
		else:
			children = [
				self._node_to_dict(ch, max_depth=max_depth, _depth=_depth + 1)
				for ch in getattr(node, "children", [])
			]

		return {
			"title": title,
			"rows": rows,
			"children": children
		}


	def _build_html(self, d3_data_json, node_w=380, node_h=140, h_gap=20, v_gap=140):
		return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Cobweb Tree</title>
<style>
	body {{ margin: 0; font-family: system-ui, sans-serif; }}
	.link {{ fill: none; stroke: #9aa1a9; stroke-width: 1.5px; }}
	.node-box {{ stroke: #444; fill: #fff; rx: 8; ry: 8; filter: drop-shadow(1px 2px 2px rgba(0,0,0,0.15)); }}
	.node-fo table {{ border-collapse: collapse; font-size: 12px; margin: 4px 0; }}
	.node-fo th, .node-fo td {{ border: 1px solid #888; padding: 2px 6px; }}
	.node-fo th {{ background: #f3f5f7; font-weight: 600; }}
	.section-title {{ font-weight: bold; margin-top: 4px; }}
	.section {{ margin-top: 10px; margin-bottom: 10px; }}
</style>
</head>
<body>
<div id="tree"></div>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>
const data = {d3_data_json};

const nodeW  = {node_w};
const nodeH  = {node_h};
const hGap   = {h_gap};
const vGap   = {v_gap};

const root = d3.hierarchy(data);
const layout = d3.tree()
	.nodeSize([nodeW + hGap, nodeH + vGap])
	.separation((a, b) => (a.parent === b.parent ? 1.0 : 1.4));
layout(root);

const svg = d3.select("#tree").append("svg")
	.attr("width", 1)
	.attr("height", 1);

const g = svg.append("g");

const linkGen = d3.linkVertical().x(d => d.x).y(d => d.y);
const link = g.selectAll("path.link")
	.data(root.links())
	.join("path")
	.attr("class", "link")
	.attr("d", linkGen);

const node = g.selectAll("g.node")
	.data(root.descendants())
	.join("g")
	.attr("transform", d => `translate(${{d.x}},${{d.y}})`);

node.append("rect")
	.attr("class", "node-box")
	.attr("x", -nodeW/2)
	.attr("y", 0)
	.attr("width", nodeW)
	.attr("height", nodeH);

node.append("foreignObject")
	.attr("class", "node-fo")
	.attr("x", -nodeW/2 + 6)
	.attr("y", 6)
	.attr("width", nodeW - 12)
	.attr("height", 1000)
	.html(d => nodeHTML(d.data));

node.selectAll("foreignObject").each(function(d) {{
	const fo = d3.select(this);
	const div = fo.select("div").node();
	const h = div.getBoundingClientRect().height + 12;
	d._nodeHeight = h;
	fo.attr("height", h);
	d3.select(this.parentNode).select("rect").attr("height", h + 12);
}});

const depthMaxHeight = new Map();
root.each(d => {{
	const h = d._nodeHeight || nodeH;
	const existing = depthMaxHeight.get(d.depth) || 0;
	if (h > existing) {{ depthMaxHeight.set(d.depth, h); }}
}});

const depthOffsets = [];
const maxDepth = Math.max(...Array.from(depthMaxHeight.keys()));
for (let i = 0; i <= maxDepth; i++) {{
	const prev = i === 0 ? 0 : depthOffsets[i - 1] + depthMaxHeight.get(i - 1) + vGap;
	depthOffsets.push(prev);
}}

root.each(d => {{ d.y = depthOffsets[d.depth]; }});

node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
link.attr("d", linkGen);

let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
root.each(d => {{
	const halfW = nodeW / 2;
	const h = d._nodeHeight || nodeH;
	if (d.x - halfW - 30 < x0) x0 = d.x - halfW - 30;
	if (d.x + halfW + 30 > x1) x1 = d.x + halfW + 30;
	if (d.y - 30 < y0) y0 = d.y - 30;
	if (d.y + h + 30 > y1) y1 = d.y + h + 30;
}});
const width  = x1 - x0;
const height = y1 - y0;

svg.attr("width", width)
	.attr("height", height)
	.attr("viewBox", [x0, y0, width, height].join(" "));

function nodeHTML(d) {{
	let tableRows = d.rows.map(r =>
		`<tr><td>${{r.attr}}</td><td>${{r.val}}</td><td>${{r.count}}</td></tr>`
	).join("");
	return `
	<div class="node-fo">
		<table>
			<tr><th colspan="3">${{d.title}}</th></tr>
			<tr><th>Attr</th><th>Value</th><th>Count</th></tr>
			${{tableRows}}
		</table>
	</div>`;
}}
</script>
</body>
</html>
"""

	async def _html_to_png(self, html_file, png_file, viewport_width=1600, viewport_height=1200):
		"""
		Convert an HTML file into a PNG screenshot using Playwright.
		"""
		async with async_playwright() as p:
				browser = await p.chromium.launch(headless=True)
				page = await browser.new_page()
				await page.set_viewport_size({"width": viewport_width, "height": viewport_height})

				url = "file://" + os.path.abspath(html_file)
				await page.goto(url)

				# wait until tree SVG is rendered
				await page.wait_for_selector("#tree svg")

				# Take full-page screenshot
				await page.screenshot(path=png_file, full_page=True)
				await browser.close()

	def draw_tree(self, root, filepath, max_depth=None):
		"""
		Draw Cobweb tree from root node and save HTML/PNG.
		"""
		d3_json = json.dumps(self._node_to_dict(root, max_depth=max_depth))
		html_str = self._build_html(d3_json)

		os.makedirs(os.path.dirname(filepath + ".html"), exist_ok=True)
		os.makedirs(os.path.dirname(filepath + ".png"), exist_ok=True)

		with open(filepath + ".html", "w", encoding="utf-8") as f:
				f.write(html_str)

		if filepath + ".png":
				asyncio.run(self._html_to_png(filepath + ".html", filepath + ".png"))

		return filepath + ".html", filepath + ".png"
	
	def save_level_subtrees(self, root, folder, level=3):
		"""
		Helper function to draw all subtrees at a certain level!

		Need to save and reload IDs as they occur at the top level!
		"""

		if os.path.exists(folder):
			try:
				shutil.rmtree(folder)
			except OSError as e:
				print(f"Error deleting folder '{folder}': {e}")

		optimal_depth = level

		leaves = []
		visited = [(0, root)]

		num_nodes = 0

		while len(visited) > 0:
			depth, curr = visited.pop()
			num_nodes += 1

			if len(curr.children) == 0:
				leaves.append((depth, curr))
			else:
				for child in curr.children:
					visited.append((depth + 1, child))

		# print("average leaf depth:", sum([x[0] for x in leaves]) / len(leaves))
		# print("median leaf depth:", statistics.median([x[0] for x in leaves]))
		
		basic_level_nodes = {}

		for leaf_tup in leaves:
			leaf_depth, leaf_node = leaf_tup
			curr_depth = leaf_depth
			curr_node = leaf_node
			while curr_depth > optimal_depth:
				curr_node = curr_node.parent
				curr_depth -= 1

			if curr_depth == optimal_depth and leaf_depth - curr_depth >= 2: # 2 is an edge case for "big enough" leaves
				basic_level_nodes[curr_node.concept_hash()] = curr_node

		# print("num nodes:", num_nodes)
		# print("num leaves:", len(leaves))
		# print(f"num nodes at depth {level}:", len(basic_level_nodes))

		for key, bl_node in basic_level_nodes.items():
			self.draw_tree(bl_node, folder + ("/" if folder[-1] != "/" else "") + f"level_{level}_{key}", max_depth=3)

		return True
	
	def save_basic_level_subtrees(self, root, folder, debug=False):
		"""
		Helper function to draw all basic-level subtrees!

		Need to save and reload IDs as they occur at the top level!
		"""

		if os.path.exists(folder):
			try:
				shutil.rmtree(folder)
			except OSError as e:
				print(f"Error deleting folder '{folder}': {e}")

		leaves = []
		visited = [(0, root)]

		num_nodes = 0

		while len(visited) > 0:
			depth, curr = visited.pop()
			num_nodes += 1

			if len(curr.children) == 0:
				leaves.append((depth, curr))
			else:
				for child in curr.children:
					visited.append((depth + 1, child))

		if debug:
			print("median leaf depth:", statistics.median([x[0] for x in leaves]))
			print("mean leaf depth:", statistics.mean([x[0] for x in leaves]))
			print("mode leaf depth:", statistics.mode([x[0] for x in leaves]), f"with count of {[x[0] for x in leaves].count(statistics.mode([x[0] for x in leaves]))} leaves")
			print("min leaf depth:", min([x[0] for x in leaves]))
			print("max leaf depth:", max([x[0] for x in leaves]))
		
		basic_level_nodes = {}
		basic_level_count = {}

		for leaf_tup in leaves:
			_, leaf_node = leaf_tup
			curr_node = leaf_node.get_basic(1000, 1000)
			if curr_node.concept_hash() != leaf_node.concept_hash():
				basic_level_nodes[curr_node.concept_hash()] = curr_node
				basic_level_count[curr_node.concept_hash()] = basic_level_count.setdefault(curr_node.concept_hash(), 1)
				basic_level_count[curr_node.concept_hash()] += 1
			else:
				# print(f"BASIC LEVEL NODE IS LEAF NODE FOR CONCEPT HASH {leaf_node.concept_hash()}")
				pass

		basic_level_count = dict(sorted(basic_level_count.items()))

		if debug:
			print("num nodes:", num_nodes)
			print("num leaves:", len(leaves))
			print(f"num nodes at basic level:", len(basic_level_nodes))

			for key, cnt in basic_level_count.items():
				print(f"Node {key} is basic-level for {cnt} nodes, has: !")
				print(f"- Entropy of {basic_level_nodes[key].entropy()}")
				print(f"- Category Utility of {basic_level_nodes[key].category_utility()}")
				print(f"- Partition Utility of {basic_level_nodes[key].partition_utility()}")


		for key, bl_node in basic_level_nodes.items():
			self.draw_tree(bl_node, folder + ("/" if folder[-1] != "/" else "") + f"basic_level_{key}", max_depth=4)

		return True
