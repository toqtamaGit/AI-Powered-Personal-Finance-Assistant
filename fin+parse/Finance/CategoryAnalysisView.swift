// CategoryAnalysisView.swift — Spending breakdown by category.

import SwiftUI
import Charts

struct CategoryAnalysisPage: View {
    @ObservedObject var service: DashboardService

    var body: some View {
        VStack(spacing: 16) {
            guard let data = service.categories, !data.categories.isEmpty else {
                return AnyView(
                    Text("No category data")
                        .font(.system(size: 14))
                        .foregroundColor(AppTheme.textMuted)
                        .frame(maxWidth: .infinity, minHeight: 200)
                )
            }

            return AnyView(content(data))
        }
    }

    @ViewBuilder
    private func content(_ data: APICategoryResponse) -> some View {
        // Total spent badge
        HStack {
            Text("Total Spent")
                .font(.system(size: 13, weight: .medium))
                .foregroundColor(AppTheme.textMuted)
            Spacer()
            Text(formatTenge(data.total))
                .font(.system(size: 17, weight: .bold))
                .foregroundColor(AppTheme.red)
        }
        .glassCard(padding: 14)
        .padding(.horizontal, 20)

        // Donut chart
        VStack(alignment: .leading, spacing: 12) {
            Text("Category Split")
                .font(.system(size: 15, weight: .bold))
                .foregroundColor(AppTheme.textPrimary)

            Chart {
                ForEach(data.categories) { c in
                    SectorMark(
                        angle: .value("Amount", c.amount),
                        innerRadius: .ratio(0.55),
                        angularInset: 1.5
                    )
                    .foregroundStyle(c.appCategory.color)
                    .annotation(position: .overlay) {
                        if c.percentage >= 5 {
                            Text("\(Int(c.percentage))%")
                                .font(.system(size: 10, weight: .bold))
                                .foregroundColor(.white)
                        }
                    }
                }
            }
            .frame(height: 200)

            FlowLegend(items: data.categories.map { ($0.category, $0.appCategory.color, $0.percentage) })
        }
        .glassCard()
        .padding(.horizontal, 20)

        // Horizontal bar chart
        VStack(alignment: .leading, spacing: 12) {
            Text("Ranked by Amount")
                .font(.system(size: 15, weight: .bold))
                .foregroundColor(AppTheme.textPrimary)

            Chart {
                ForEach(data.categories) { c in
                    BarMark(
                        x: .value("Amount", c.amount),
                        y: .value("Category", c.category)
                    )
                    .foregroundStyle(c.appCategory.color)
                    .annotation(position: .trailing, alignment: .leading) {
                        Text(shortTenge(c.amount))
                            .font(.system(size: 10))
                            .foregroundColor(AppTheme.textMuted)
                    }
                }
            }
            .chartXAxis(.hidden)
            .frame(height: CGFloat(data.categories.count) * 44)
        }
        .glassCard()
        .padding(.horizontal, 20)

        // Detail rows with insights
        VStack(alignment: .leading, spacing: 8) {
            Text("Insights")
                .font(.system(size: 15, weight: .bold))
                .foregroundColor(AppTheme.textPrimary)

            ForEach(data.categories) { c in
                HStack(spacing: 10) {
                    Image(systemName: c.appCategory.icon)
                        .font(.system(size: 14))
                        .foregroundColor(.white)
                        .frame(width: 30, height: 30)
                        .background(c.appCategory.color)
                        .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))

                    VStack(alignment: .leading, spacing: 2) {
                        Text(c.category)
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundColor(AppTheme.textPrimary)
                        Text("You spent \(Int(c.percentage))% on \(c.category.lowercased()) (\(c.count) transactions)")
                            .font(.system(size: 11))
                            .foregroundColor(AppTheme.textMuted)
                    }

                    Spacer()

                    Text(formatTenge(c.amount))
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundColor(AppTheme.textPrimary)
                }
                .padding(.vertical, 4)

                if c.id != data.categories.last?.id {
                    Divider()
                }
            }
        }
        .glassCard()
        .padding(.horizontal, 20)
    }
}
