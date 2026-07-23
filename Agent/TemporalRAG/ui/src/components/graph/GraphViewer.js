import React, { useEffect, useRef, useCallback, useState } from 'react';
import * as d3 from 'd3';

const NODE_COLORS = {
  Company: '#3b82f6',
  Article: '#10b981',
  Theme:   '#8b5cf6',
};

const NODE_RADII = {
  Company: (degree) => Math.max(12, Math.min(36, 12 + Math.sqrt(Math.max(degree || 0, 0)) * 2.5)),
  Article: () => 7,
  Theme:   () => 6,
};

const EDGE_COLORS = {
  CO_OCCURRED_WITH: 'rgba(59,130,246,0.5)',
  MENTIONED_IN:     'rgba(16,185,129,0.4)',
  HAS_THEME:        'rgba(139,92,246,0.4)',
};

const GraphViewer = ({ data, activeTypes, onNodeClick, expandedTicker, width = '100%', height = 600 }) => {
  const svgRef = useRef(null);
  const simRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);

  const draw = useCallback(() => {
    if (!svgRef.current || !data) return;

    const el = svgRef.current;
    const W = el.parentElement?.clientWidth || 800;
    const H = height;

    // Filter: show all node types that exist, but respect activeTypes for Article/Theme
    // Company is always shown; Article & Theme follow activeTypes
    const visibleNodes = data.nodes.filter(n => {
      if (n.type === 'Company') return true;
      return activeTypes[n.type];
    });
    const visibleNodeIds = new Set(visibleNodes.map(n => n.id));
    const visibleEdges = data.edges.filter(e => {
      const src = e.source?.id ?? e.source;
      const tgt = e.target?.id ?? e.target;
      return visibleNodeIds.has(src) && visibleNodeIds.has(tgt);
    });

    // Stop existing simulation
    if (simRef.current) simRef.current.stop();
    d3.select(el).selectAll('*').remove();

    const svg = d3.select(el)
      .attr('width', W)
      .attr('height', H)
      .attr('viewBox', `0 0 ${W} ${H}`);

    // Background
    svg.append('rect')
      .attr('width', W).attr('height', H)
      .attr('fill', 'transparent');

    // Zoom & Pan
    const g = svg.append('g');
    const zoom = d3.zoom()
      .scaleExtent([0.05, 6])
      .on('zoom', e => g.attr('transform', e.transform));
    svg.call(zoom);

    // Defs: arrowheads + glow filters
    const defs = svg.append('defs');

    // Glow filter
    const filter = defs.append('filter').attr('id', 'glow').attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
    filter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'blur');
    const feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'blur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Arrowheads per edge type
    Object.entries(EDGE_COLORS).forEach(([type, color]) => {
      defs.append('marker')
        .attr('id', `arrow-${type}`)
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 28).attr('refY', 0)
        .attr('markerWidth', 4).attr('markerHeight', 4)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', color);
    });

    // Edge weight scale
    const maxWeight = d3.max(visibleEdges, e => e.weight || 1) || 1;
    const edgeWidthScale = d3.scaleLinear().domain([1, maxWeight]).range([0.6, 4]);

    // Clone nodes/edges so d3 can mutate x/y/vx/vy freely
    const nodes = visibleNodes.map(n => ({ ...n }));
    const nodeById = new Map(nodes.map(n => [n.id, n]));
    const edges = visibleEdges.map(e => ({
      ...e,
      source: nodeById.get(e.source?.id ?? e.source) ?? e.source,
      target: nodeById.get(e.target?.id ?? e.target) ?? e.target,
    }));

    // Force simulation — tuned parameters
    const companyCount = nodes.filter(n => n.type === 'Company').length;
    const chargeStrength = companyCount > 20 ? -400 : -300;

    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(edges)
        .id(d => d.id)
        .distance(d => {
          if (d.type === 'CO_OCCURRED_WITH') return 120;
          if (d.type === 'MENTIONED_IN')     return 70;
          return 50;
        })
        .strength(d => {
          if (d.type === 'CO_OCCURRED_WITH') return 0.4;
          return 0.6;
        }))
      .force('charge', d3.forceManyBody().strength(chargeStrength).distanceMax(400))
      .force('center', d3.forceCenter(W / 2, H / 2).strength(0.05))
      .force('collision', d3.forceCollide(d => (NODE_RADII[d.type] || (() => 8))(d.degree) + 8).strength(0.9))
      .alphaDecay(0.025)
      .velocityDecay(0.4);

    simRef.current = sim;

    // ── Draw edges ──
    const linkGroup = g.append('g').attr('class', 'links');
    const link = linkGroup.selectAll('line')
      .data(edges).join('line')
      .attr('stroke', d => EDGE_COLORS[d.type] || 'rgba(255,255,255,0.15)')
      .attr('stroke-width', d => d.weight ? edgeWidthScale(d.weight) : 0.8)
      .attr('stroke-opacity', d => d.type === 'CO_OCCURRED_WITH' ? 0.6 : 0.4)
      .attr('marker-end', d => `url(#arrow-${d.type})`);

    // ── Draw nodes ──
    const nodeGroup = g.append('g').attr('class', 'nodes');
    const node = nodeGroup.selectAll('g')
      .data(nodes).join('g')
      .attr('class', 'node-g')
      .attr('cursor', d => d.type === 'Company' ? 'pointer' : 'default')
      .call(d3.drag()
        .on('start', (event, d) => {
          if (!event.active) sim.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x; d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) sim.alphaTarget(0);
          d.fx = null; d.fy = null;
        })
      );

    // Outer glow ring for expanded node
    node.filter(d => d.type === 'Company').append('circle')
      .attr('class', 'node-ring')
      .attr('r', d => (NODE_RADII.Company(d.degree) + 5))
      .attr('fill', 'none')
      .attr('stroke', d => d.id === expandedTicker ? '#06b6d4' : NODE_COLORS.Company)
      .attr('stroke-width', d => d.id === expandedTicker ? 2 : 0)
      .attr('stroke-opacity', d => d.id === expandedTicker ? 0.8 : 0)
      .attr('stroke-dasharray', '4 2');

    // Main circle
    node.append('circle')
      .attr('class', 'node-circle')
      .attr('r', d => (NODE_RADII[d.type] || (() => 8))(d.degree))
      .attr('fill', d => NODE_COLORS[d.type] || '#666')
      .attr('fill-opacity', d => d.type === 'Company' ? 0.9 : 0.75)
      .attr('stroke', d => NODE_COLORS[d.type] || '#666')
      .attr('stroke-width', d => d.type === 'Company' ? 2 : 1)
      .attr('stroke-opacity', 0.8)
      .attr('filter', d => d.type === 'Company' ? 'url(#glow)' : 'none');

    // Labels — Company ticker
    node.filter(d => d.type === 'Company').append('text')
      .attr('class', 'node-label')
      .attr('dy', d => NODE_RADII.Company(d.degree) + 13)
      .attr('text-anchor', 'middle')
      .attr('font-size', 10)
      .attr('font-weight', 600)
      .attr('font-family', 'Inter, sans-serif')
      .attr('fill', '#94a3b8')
      .attr('pointer-events', 'none')
      .text(d => d.ticker || d.id);

    // ── Interactions ──
    node
      .on('mouseenter', function(event, d) {
        // Highlight: scale up circle
        d3.select(this).select('.node-circle')
          .transition().duration(120)
          .attr('r', (NODE_RADII[d.type] || (() => 8))(d.degree) * 1.25)
          .attr('fill-opacity', 1)
          .attr('stroke-width', d.type === 'Company' ? 3 : 2);

        // Show ring
        if (d.type === 'Company') {
          d3.select(this).select('.node-ring')
            .transition().duration(120)
            .attr('stroke-width', 2)
            .attr('stroke-opacity', 0.6);
        }

        // Highlight connected edges
        const nodeId = d.id;
        link.transition().duration(100)
          .attr('stroke-opacity', e => {
            const src = e.source?.id ?? e.source;
            const tgt = e.target?.id ?? e.target;
            return (src === nodeId || tgt === nodeId) ? 1 : 0.1;
          })
          .attr('stroke-width', e => {
            const src = e.source?.id ?? e.source;
            const tgt = e.target?.id ?? e.target;
            return (src === nodeId || tgt === nodeId)
              ? (e.weight ? edgeWidthScale(e.weight) * 2 : 2)
              : (e.weight ? edgeWidthScale(e.weight) : 0.8);
          });

        setTooltip({ x: event.clientX, y: event.clientY, node: d });
      })
      .on('mousemove', (event) => {
        setTooltip(prev => prev ? { ...prev, x: event.clientX, y: event.clientY } : null);
      })
      .on('mouseleave', function(event, d) {
        d3.select(this).select('.node-circle')
          .transition().duration(150)
          .attr('r', (NODE_RADII[d.type] || (() => 8))(d.degree))
          .attr('fill-opacity', d.type === 'Company' ? 0.9 : 0.75)
          .attr('stroke-width', d.type === 'Company' ? 2 : 1);

        if (d.type === 'Company') {
          d3.select(this).select('.node-ring')
            .transition().duration(150)
            .attr('stroke-width', d.id === expandedTicker ? 2 : 0)
            .attr('stroke-opacity', d.id === expandedTicker ? 0.8 : 0);
        }

        // Restore edges
        link.transition().duration(150)
          .attr('stroke-opacity', e => e.type === 'CO_OCCURRED_WITH' ? 0.6 : 0.4)
          .attr('stroke-width', e => e.weight ? edgeWidthScale(e.weight) : 0.8);

        setTooltip(null);
      })
      .on('click', (event, d) => {
        event.stopPropagation();
        if (d.type !== 'Company') return;
        setSelectedNode(prev => prev?.id === d.id ? null : d);
        if (onNodeClick) onNodeClick(d);
      });

    // Click on background → deselect
    svg.on('click', () => {
      setSelectedNode(null);
    });

    // ── Tick ──
    sim.on('tick', () => {
      // Constrain nodes within bounds
      nodes.forEach(d => {
        const r = (NODE_RADII[d.type] || (() => 8))(d.degree) + 4;
        d.x = Math.max(r, Math.min(W - r, d.x || W / 2));
        d.y = Math.max(r, Math.min(H - r, d.y || H / 2));
      });

      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => {
          const dx = d.target.x - d.source.x;
          const dy = d.target.y - d.source.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const r = (NODE_RADII[d.target.type] || (() => 8))(d.target.degree);
          return d.target.x - (dx / dist) * r;
        })
        .attr('y2', d => {
          const dx = d.target.x - d.source.x;
          const dy = d.target.y - d.source.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const r = (NODE_RADII[d.target.type] || (() => 8))(d.target.degree);
          return d.target.y - (dy / dist) * r;
        });

      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    return () => sim.stop();
  }, [data, activeTypes, height, onNodeClick, expandedTicker]);

  useEffect(() => {
    const cleanup = draw();
    return () => {
      if (cleanup) cleanup();
      simRef.current?.stop();
    };
  }, [draw]);

  return (
    <div style={{ position: 'relative', width, height }}>
      <svg
        ref={svgRef}
        style={{ width: '100%', height: '100%', background: 'transparent' }}
      />

      {/* Tooltip */}
      {tooltip && (
        <div
          className="graph-tooltip"
          style={{ left: tooltip.x + 14, top: tooltip.y - 14 }}
        >
          <div className="graph-tooltip-header">
            <span
              className="graph-tooltip-dot"
              style={{ background: NODE_COLORS[tooltip.node.type] || '#666' }}
            />
            <span style={{ fontWeight: 700, color: NODE_COLORS[tooltip.node.type] || 'white' }}>
              {tooltip.node.ticker || tooltip.node.id}
            </span>
            <span className="graph-tooltip-type">{tooltip.node.type}</span>
          </div>

          {tooltip.node.name && tooltip.node.name !== tooltip.node.ticker && (
            <div className="graph-tooltip-name">{tooltip.node.name}</div>
          )}
          {tooltip.node.label && tooltip.node.type !== 'Company' && (
            <div className="graph-tooltip-name">{tooltip.node.label}</div>
          )}
          {tooltip.node.degree != null && (
            <div className="graph-tooltip-row">
              <span className="graph-tooltip-key">Connections</span>
              <span className="graph-tooltip-val">{Math.round(tooltip.node.degree)}</span>
            </div>
          )}
          {tooltip.node.published_date && (
            <div className="graph-tooltip-row">
              <span className="graph-tooltip-key">Published</span>
              <span className="graph-tooltip-val">{tooltip.node.published_date}</span>
            </div>
          )}
          {tooltip.node.type === 'Company' && (
            <div className="graph-tooltip-action">
              {selectedNode?.id === tooltip.node.id ? '▼ Click to collapse' : '▶ Click to expand'}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default GraphViewer;
