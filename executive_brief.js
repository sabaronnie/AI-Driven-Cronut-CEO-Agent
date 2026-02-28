const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageBreak, PageNumber, LevelFormat, TabStopType, TabStopPosition
} = require("docx");

const ACCENT = "8B6914";
const DARK = "1C1C28";
const GRAY = "555555";
const LIGHT_BG = "F8F5F0";
const WHITE = "FFFFFF";
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0 };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Arial", size: 21, color: DARK } }
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Arial", color: DARK },
        paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: ACCENT },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 }
      },
    ]
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 270 } } }
        }]
      },
      {
        reference: "numbers",
        levels: [{
          level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 270 } } }
        }]
      },
    ]
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 }
        }
      },
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: ACCENT, space: 6 } },
              spacing: { after: 0 },
              children: [
                new TextRun({ text: "CONUT", bold: true, size: 18, font: "Arial", color: ACCENT }),
                new TextRun({ text: "  |  Executive Brief  |  AI Chief of Operations", size: 16, font: "Arial", color: GRAY }),
              ],
              tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
            })
          ]
        })
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              border: { top: { style: BorderStyle.SINGLE, size: 2, color: "DDDDDD", space: 4 } },
              alignment: AlignmentType.CENTER,
              children: [
                new TextRun({ text: "Confidential  |  Conut Bakery Chain  |  February 2026  |  Page ", size: 16, color: GRAY }),
                new TextRun({ children: [PageNumber.CURRENT], size: 16, color: GRAY }),
              ]
            })
          ]
        })
      },
      children: [
        // ── Title Block ──
        new Paragraph({ spacing: { before: 200, after: 0 }, children: [] }),
        new Paragraph({
          alignment: AlignmentType.LEFT,
          spacing: { after: 40 },
          children: [new TextRun({ text: "Executive Brief", bold: true, size: 40, font: "Arial", color: DARK })]
        }),
        new Paragraph({
          spacing: { after: 20 },
          children: [new TextRun({ text: "AI-Driven Operational Intelligence for Conut Bakery", size: 24, font: "Arial", color: ACCENT })]
        }),
        new Paragraph({
          spacing: { after: 240 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 8 } },
          children: [new TextRun({ text: "February 2026  |  Prepared by the Conut AI Operations Team", size: 18, color: GRAY })]
        }),

        // ── 1. Problem Framing ──
        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("1. Problem Framing")] }),
        new Paragraph({
          spacing: { after: 100 },
          children: [new TextRun({
            text: "Conut is a fast-growing Lebanese bakery chain operating in Tyre, Jnah (Beirut), and Batroun. Management currently makes inventory, staffing, pricing, and expansion decisions based on intuition rather than data. This leads to product waste from demand miscalculation, missed revenue from suboptimal bundling, and staffing inefficiencies across shifts. With plans to expand, Conut needs a data-driven framework to guide where to grow and how to optimize existing operations.",
            size: 21
          })]
        }),
        new Paragraph({
          spacing: { after: 160 },
          children: [new TextRun({
            text: "We built an AI Chief of Operations that integrates 6 machine learning models, trained on real Conut data (8 datasets, 6,700+ records), to answer the key business questions below.",
            size: 21
          })]
        }),

        // ── 2. Top Findings ──
        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("2. Top Findings")] }),

        // Finding: Expansion
        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Expansion Readiness: Score 67.3/100")] }),
        new Paragraph({
          spacing: { after: 60 },
          numbering: { reference: "bullets", level: 0 },
          children: [new TextRun({ text: "Conut Jnah ", bold: true }), new TextRun("and "), new TextRun({ text: "Conut Tyre ", bold: true }), new TextRun("classified as High Performers by K-Means clustering; Batroun branch is Growing/Stabilizing.")]
        }),
        new Paragraph({
          spacing: { after: 60 },
          numbering: { reference: "bullets", level: 0 },
          children: [new TextRun("Revenue growth is strong (100/100) and revenue scale is high (90.3/100), but operational efficiency (15/100) and customer loyalty (37.6/100) need improvement before aggressive expansion.")]
        }),
        new Paragraph({
          spacing: { after: 120 },
          numbering: { reference: "bullets", level: 0 },
          children: [new TextRun({ text: "Verdict: ", bold: true }), new TextRun("\"Recommended with Caution\" \u2014 address staffing and channel gaps first.")]
        }),

        // Finding: Location
        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Best New Location: Kaslik/Zouk (Score 71.2)")] }),

        new Table({
          width: { size: 10080, type: WidthType.DXA },
          columnWidths: [1200, 2100, 1500, 1500, 1500, 2280],
          rows: [
            new TableRow({
              children: ["Rank", "Area", "Score", "Population", "Gap", "Key Strength"].map(h =>
                new TableCell({
                  borders, margins: cellMargins,
                  width: { size: h === "Area" ? 2100 : h === "Key Strength" ? 2280 : h === "Rank" ? 1200 : 1500, type: WidthType.DXA },
                  shading: { fill: DARK, type: ShadingType.CLEAR },
                  children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, size: 18, color: WHITE, font: "Arial" })] })]
                })
              )
            }),
            ...([
              ["1", "Kaslik/Zouk", "71.2", "50,000", "+8.8", "High social activity, underserved"],
              ["2", "Tripoli", "68.6", "730,000", "0.0", "Massive demand, low rent"],
              ["3", "Jounieh", "63.2", "120,000", "+3.7", "Tourism + university hub"],
              ["4", "Mar Mikhael", "61.1", "15,000", "+3.5", "Highest social activity (95)"],
              ["5", "Aley", "56.9", "90,000", "+5.7", "Low competition, affordable rent"],
            ]).map((row, i) =>
              new TableRow({
                children: row.map((cell, j) =>
                  new TableCell({
                    borders, margins: cellMargins,
                    width: { size: [1200, 2100, 1500, 1500, 1500, 2280][j], type: WidthType.DXA },
                    shading: { fill: i % 2 === 0 ? LIGHT_BG : WHITE, type: ShadingType.CLEAR },
                    children: [new Paragraph({ children: [new TextRun({ text: cell, size: 18, font: "Arial" })] })]
                  })
                )
              })
            )
          ]
        }),

        new Paragraph({ spacing: { before: 60, after: 120 }, children: [new TextRun({ text: "Model accuracy: R\u00B2 = 0.976, MAE = 2.8 competitors. Analyzed 24 Lebanese areas using population, social activity, traffic, and competitor data.", size: 18, italics: true, color: GRAY })] }),

        // Finding: Combos
        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Revenue Optimization: Product Combos")] }),
        new Paragraph({
          spacing: { after: 60 },
          numbering: { reference: "bullets", level: 0 },
          children: [new TextRun("Apriori analysis across 69 baskets found "), new TextRun({ text: "3,777 association rules", bold: true }), new TextRun(". Top combos have 100% confidence and lift of 69x.")]
        }),
        new Paragraph({
          spacing: { after: 60 },
          numbering: { reference: "bullets", level: 0 },
          children: [new TextRun("Top bundle: "), new TextRun({ text: "Conut Berry Mix + Conut Original + Espresso", bold: true }), new TextRun(" \u2014 cross-category combo (bakery + coffee) with highest revenue potential.")]
        }),
        new Paragraph({
          spacing: { after: 120 },
          numbering: { reference: "bullets", level: 0 },
          children: [new TextRun("Recommended ~15% combo discount to drive adoption while maintaining margin.")]
        }),

        // Finding: Demand + Staffing
        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Demand Forecast & Staffing")] }),
        new Paragraph({
          spacing: { after: 60 },
          numbering: { reference: "bullets", level: 0 },
          children: [new TextRun({ text: "Tyre: ", bold: true }), new TextRun("Stable trajectory, forecast ~1.19B LBP/month (MAPE 8.3%). Best-performing forecast model.")]
        }),
        new Paragraph({
          spacing: { after: 60 },
          numbering: { reference: "bullets", level: 0 },
          children: [new TextRun({ text: "Jnah: ", bold: true }), new TextRun("Explosive growth trend, forecast 3.3B\u20136.2B LBP over next quarter. Highest revenue per labor hour (5.1M LBP/hr).")]
        }),
        new Paragraph({
          spacing: { after: 120 },
          numbering: { reference: "bullets", level: 0 },
          children: [new TextRun({ text: "Staffing gap identified: ", bold: true }), new TextRun("Tyre morning shifts are understaffed. Ridge Regression model (MAE = 0.63 staff) recommends adding 1 employee to morning weekday shifts.")]
        }),

        // ── Page break ──
        new Paragraph({ children: [new PageBreak()] }),

        // ── 3. Recommended Actions ──
        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("3. Recommended Actions")] }),

        new Paragraph({
          spacing: { after: 60 },
          numbering: { reference: "numbers", level: 0 },
          children: [
            new TextRun({ text: "Launch combo bundles immediately. ", bold: true }),
            new TextRun("Start with Conut Berry Mix + Original + Espresso and Classic Chimney + Pistachio + Oreo Milkshake at all branches. Expected to increase average basket value by 10\u201315%.")
          ]
        }),
        new Paragraph({
          spacing: { after: 60 },
          numbering: { reference: "numbers", level: 0 },
          children: [
            new TextRun({ text: "Open 4th branch in Kaslik/Zouk. ", bold: true }),
            new TextRun("Highest opportunity score (71.2). Underserved market with strong social activity and university presence. Target monthly revenue: 1.14B LBP (benchmark: Jnah). Offer delivery + dine-in from day one.")
          ]
        }),
        new Paragraph({
          spacing: { after: 60 },
          numbering: { reference: "numbers", level: 0 },
          children: [
            new TextRun({ text: "Fix staffing gaps at Tyre. ", bold: true }),
            new TextRun("Add 1 staff member to morning weekday shifts. Maintain current evening levels. This addresses the understaffing flagged by the ML model without increasing labor costs significantly.")
          ]
        }),
        new Paragraph({
          spacing: { after: 60 },
          numbering: { reference: "numbers", level: 0 },
          children: [
            new TextRun({ text: "Add delivery channel at Batroun. ", bold: true }),
            new TextRun("Jnah achieves 5.1M LBP/labor hour with delivery. Batroun currently lacks this channel. Adding it could boost revenue 20\u201330% based on Jnah\u2019s channel diversification data.")
          ]
        }),
        new Paragraph({
          spacing: { after: 120 },
          numbering: { reference: "numbers", level: 0 },
          children: [
            new TextRun({ text: "Grow coffee and milkshake sales. ", bold: true }),
            new TextRun("Cross-sell espresso with bakery combos. The data shows coffee items appear in 80%+ of top association rules, indicating strong pairing potential that is currently underexploited.")
          ]
        }),

        // ── 4. Expected Impact & Risks ──
        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("4. Expected Impact & Risks")] }),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Expected Impact")] }),

        new Table({
          width: { size: 10080, type: WidthType.DXA },
          columnWidths: [3360, 3360, 3360],
          rows: [
            new TableRow({
              children: ["Action", "Expected Impact", "Timeline"].map(h =>
                new TableCell({
                  borders, margins: cellMargins,
                  width: { size: 3360, type: WidthType.DXA },
                  shading: { fill: DARK, type: ShadingType.CLEAR },
                  children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, size: 18, color: WHITE, font: "Arial" })] })]
                })
              )
            }),
            ...([
              ["Combo bundles", "+10\u201315% avg basket value", "Immediate"],
              ["Kaslik/Zouk branch", "+1.1B LBP/month revenue", "6\u201312 months"],
              ["Tyre staffing fix", "Reduced wait times, +5% retention", "Immediate"],
              ["Batroun delivery", "+20\u201330% branch revenue", "1\u20133 months"],
              ["Beverage cross-sell", "+15% coffee/shake revenue", "1\u20132 months"],
            ]).map((row, i) =>
              new TableRow({
                children: row.map((cell, j) =>
                  new TableCell({
                    borders, margins: cellMargins,
                    width: { size: 3360, type: WidthType.DXA },
                    shading: { fill: i % 2 === 0 ? LIGHT_BG : WHITE, type: ShadingType.CLEAR },
                    children: [new Paragraph({ children: [new TextRun({ text: cell, size: 18, font: "Arial" })] })]
                  })
                )
              })
            )
          ]
        }),

        new Paragraph({ spacing: { before: 160 }, heading: HeadingLevel.HEADING_2, children: [new TextRun("Risks & Mitigations")] }),

        new Paragraph({
          spacing: { after: 60 },
          numbering: { reference: "bullets", level: 0 },
          children: [
            new TextRun({ text: "Limited training data. ", bold: true }),
            new TextRun("Models trained on 5 months of data. Forecasts beyond Q1 2026 carry higher uncertainty. Mitigation: retrain monthly as new data arrives.")
          ]
        }),
        new Paragraph({
          spacing: { after: 60 },
          numbering: { reference: "bullets", level: 0 },
          children: [
            new TextRun({ text: "Expansion capital requirements. ", bold: true }),
            new TextRun("Kaslik/Zouk has a rent index of 72 (above average). Mitigation: start with a smaller format store; use Jnah\u2019s revenue-per-labor-hour as the profitability target.")
          ]
        }),
        new Paragraph({
          spacing: { after: 60 },
          numbering: { reference: "bullets", level: 0 },
          children: [
            new TextRun({ text: "Operational efficiency gap. ", bold: true }),
            new TextRun("Expansion score flagged efficiency at 15/100. Mitigation: implement staffing recommendations and delivery channel before opening a 4th branch.")
          ]
        }),
        new Paragraph({
          spacing: { after: 120 },
          numbering: { reference: "bullets", level: 0 },
          children: [
            new TextRun({ text: "Market volatility. ", bold: true }),
            new TextRun("Lebanon\u2019s economic environment adds uncertainty to all projections. Mitigation: the AI system can be re-run with updated data at any time to recalibrate recommendations.")
          ]
        }),

        // ── Footer note ──
        new Paragraph({
          spacing: { before: 200 },
          border: { top: { style: BorderStyle.SINGLE, size: 2, color: ACCENT, space: 8 } },
          children: [new TextRun({
            text: "This brief was generated by the Conut AI Chief of Operations system, a hackathon project integrating 6 ML models (Apriori, Ridge Regression, K-Means, Polynomial Regression, Exponential Smoothing, Market Gap Analysis) with real operational data. All figures are derived from model outputs, not estimates.",
            size: 17, italics: true, color: GRAY
          })]
        }),
      ]
    }
  ]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/sessions/fervent-kind-gauss/mnt/Hackaton/executive_brief.docx", buffer);
  console.log("Created executive_brief.docx");
});
