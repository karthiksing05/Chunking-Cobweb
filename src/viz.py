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
	def __init__(self, attributes, id_to_value, value_to_id):
		self.id_to_attr = attributes
		self.attr_to_id = {w: i for i, w in enumerate(attributes)}
		self.id_to_value = id_to_value
		self.value_to_id = value_to_id

	def _safe_lookup(self, id_to_list, idx):
		return id_to_list[idx] if (idx is not None and 0 <= idx < len(id_to_list)) else "None"

	def _node_to_dict(self, node, max_depth=None, _depth=0):
		"""
		Convert a CobwebNode into a JSON dict for D3 rendering.

		Args:
			node: CobwebNode
			max_depth: maximum depth to recurse (int or None).
					Depth 0 means only the current node,
					depth=1 includes children, etc.
			_depth: internal recursion counter.
		"""
		title = f"CONCEPT-{node.concept_hash()}"

		attr_tables = []
		for attr_id, val_counts in sorted(node.av_count.items()):
			attr_name = self._safe_lookup(self.id_to_attr, attr_id)

			# Sort values by descending count, then take top 7
			top_vals = sorted(val_counts.items(), key=lambda x: x[1], reverse=True)[:7]

			rows = []
			for val_id, count in top_vals:
				val_name = self._safe_lookup(self.id_to_value, val_id)
				rows.append({"val": val_name, "count": count})

			attr_tables.append({
				"attr": attr_name,
				"rows": rows
			})

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
			"attributes": attr_tables,
			"children": children
		}


	def _build_html(self, d3_data_json, node_w=320, node_h=140, h_gap=80, v_gap=200):
		# unchanged: your D3 tree HTML builder
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
const layout = d3.tree().nodeSize([nodeW + hGap, nodeH + vGap]);
layout(root);

// compute bounds
let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
root.each(d => {{
		if (d.x < x0) x0 = d.x;
		if (d.x > x1) x1 = d.x;
		if (d.y < y0) y0 = d.y;
		if (d.y > y1) y1 = d.y;
}});
const width  = x1 - x0 + nodeW + 320;
const height = y1 - y0 + nodeH + 320; // THIS IS THE HEIGHT MODIFIER

const svg = d3.select("#tree").append("svg")
	.attr("width", width)
	.attr("height", height)
	.attr("viewBox", [x0 - nodeW/2 - 50, y0 - nodeH/2 - 50, width, height].join(" "));

const g = svg.append("g");

// links
g.selectAll("path.link")
	.data(root.links())
	.join("path")
	.attr("class", "link")
	.attr("d", d3.linkVertical().x(d => d.x).y(d => d.y));

// nodes
const node = g.selectAll("g.node")
	.data(root.descendants())
	.join("g")
	.attr("transform", d => `translate(${{d.x}},${{d.y}})`);

// add bounding rect
node.append("rect")
	.attr("class", "node-box")
	.attr("x", -nodeW/2)
	.attr("y", 0)
	.attr("width", nodeW)
	.attr("height", nodeH);

// add HTML content
node.append("foreignObject")
	.attr("class", "node-fo")
	.attr("x", -nodeW/2 + 6)
	.attr("y", 6)
	.attr("width", nodeW - 12)
	.attr("height", 1000)
	.html(d => nodeHTML(d.data));

// resize rects to match actual content height
node.selectAll("foreignObject").each(function(d) {{
	const fo = d3.select(this);
	const div = fo.select("div").node();
	const h = div.getBoundingClientRect().height + 12; // padding
	fo.attr("height", h);
	d3.select(this.parentNode).select("rect")
		.attr("height", h + 12); // adjust rect height
}});

function nodeHTML(d) {{
	const attrTables = d.attributes.map(a => {{
		const rows = a.rows.map(r => `<tr><td>${{r.val}}</td><td>${{r.count}}</td></tr>`).join("");
		return `<div class="section"><div class="section-title">${{a.attr}}</div><table><tbody>${{rows}}</tbody></table></div>`;
	}}).join("");
	return `
	<div class="node-fo">
		<table><tr><th>${{d.title}}</th></tr></table>
		${{attrTables}}
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
			print("average leaf depth:", sum([x[0] for x in leaves]) / len(leaves))
			print("median leaf depth:", statistics.median([x[0] for x in leaves]))
		
		basic_level_nodes = {}

		for leaf_tup in leaves:
			_, leaf_node = leaf_tup
			curr_node = leaf_node.get_basic_level()
			if curr_node.concept_hash() != leaf_node.concept_hash():
				basic_level_nodes[curr_node.concept_hash()] = curr_node

		if debug:
			print("num nodes:", num_nodes)
			print("num leaves:", len(leaves))
			print(f"num nodes at basic level:", len(basic_level_nodes))

		for key, bl_node in basic_level_nodes.items():
			self.draw_tree(bl_node, folder + ("/" if folder[-1] != "/" else "") + f"basic_level_{key}", max_depth=3)

		return True
