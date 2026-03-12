// BankAnalysisView.swift — Spending comparison across banks.

import SwiftUI
import Charts

struct BankAnalysisPage: View {
    @ObservedObject var service: DashboardService

    // Stable colours for the two banks
    private let bankColors: [String: Color] = [
        "Kaspi Bank": Color(hex: "#F44336"),
        "Freedom Bank Kazakhstan": Color(hex: "#2196F3"),
    ]

    private func bankColor(_ name: String) -> Color {
        bankColors[name] ?? AppTheme.accent
    }

    var body: some View {
        VStack(spacing: 16) {
            guard let data = service.banks, !data.banks.isEmpty else {
                return AnyView(
                    Text("No bank data")
                        .font(.system(size: 14))
                        .foregroundColor(AppTheme.textMuted)
                        .frame(maxWidth: .infinity, minHeight: 200)
                )
            }

            return AnyView(content(data))
        }
    }

    @ViewBuilder
    private func content(_ data: APIBankResponse) -> some View {
        // Total
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

        // Pie chart — share per bank
        VStack(alignment: .leading, spacing: 12) {
            Text("Bank Share")
                .font(.system(size: 15, weight: .bold))
                .foregroundColor(AppTheme.textPrimary)

            Chart {
                ForEach(data.banks) { b in
                    SectorMark(
                        angle: .value("Amount", b.amount),
                        innerRadius: .ratio(0.55),
                        angularInset: 2
                    )
                    .foregroundStyle(bankColor(b.bank))
                    .annotation(position: .overlay) {
                        Text("\(Int(b.percentage))%")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundColor(.white)
                    }
                }
            }
            .frame(height: 200)

            HStack(spacing: 16) {
                ForEach(data.banks) { b in
                    LegendDot(color: bankColor(b.bank), label: b.shortName)
                }
            }
        }
        .glassCard()
        .padding(.horizontal, 20)

        // KPI cards per bank
        ForEach(data.banks) { b in
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    Circle()
                        .fill(bankColor(b.bank))
                        .frame(width: 10, height: 10)
                    Text(b.shortName)
                        .font(.system(size: 16, weight: .bold))
                        .foregroundColor(AppTheme.textPrimary)
                }

                HStack(spacing: 12) {
                    StatBox(label: "Total", value: formatTenge(b.amount))
                    StatBox(label: "Count", value: "\(b.count)")
                    StatBox(label: "Average", value: formatTenge(b.average))
                }
            }
            .glassCard()
            .padding(.horizontal, 20)
        }
    }
}

// MARK: - Stat Box
struct StatBox: View {
    let label: String
    let value: String

    var body: some View {
        VStack(spacing: 4) {
            Text(label)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(AppTheme.textMuted)
            Text(value)
                .font(.system(size: 13, weight: .bold))
                .foregroundColor(AppTheme.textPrimary)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(AppTheme.surface2)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}
