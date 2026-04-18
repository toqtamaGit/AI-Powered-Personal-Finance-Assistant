// TransactionTypeView.swift — Breakdown by operation type.

import SwiftUI
import Charts

struct TransactionTypePage: View {
    @ObservedObject var service: DashboardService

    // Unique colour per operation
    private let opColors: [String: Color] = [
        "Purchases":      Color(hex: "#EF4444"),
        "Replenishment":  Color(hex: "#10B981"),
        "Transfers":      Color(hex: "#6366F1"),
        "Transfer":       Color(hex: "#818CF8"),
        "Acquiring":      Color(hex: "#F59E0B"),
        "Withdrawal":     Color(hex: "#F97316"),
        "Others":         Color(hex: "#6B7280"),
        "Payment":        Color(hex: "#06B6D4"),
        "Pending amount": Color(hex: "#A78BFA"),
    ]

    private func opColor(_ name: String) -> Color {
        opColors[name] ?? AppTheme.textMuted
    }

    var body: some View {
        VStack(spacing: 16) {
            guard let data = service.operations, !data.operations.isEmpty else {
                return AnyView(
                    Text("No operation data")
                        .font(.system(size: 14))
                        .foregroundColor(AppTheme.textMuted)
                        .frame(maxWidth: .infinity, minHeight: 200)
                )
            }

            return AnyView(content(data))
        }
    }

    @ViewBuilder
    private func content(_ data: APIOperationResponse) -> some View {
        // Pie chart — operation distribution
        VStack(alignment: .leading, spacing: 12) {
            Text("Transaction Types")
                .font(.system(size: 15, weight: .bold))
                .foregroundColor(AppTheme.textPrimary)

            Chart {
                ForEach(data.operations) { op in
                    SectorMark(
                        angle: .value("Amount", op.amount),
                        innerRadius: .ratio(0.55),
                        angularInset: 1.5
                    )
                    .foregroundStyle(opColor(op.operation))
                    .annotation(position: .overlay) {
                        if op.percentage >= 5 {
                            Text("\(Int(op.percentage))%")
                                .font(.system(size: 10, weight: .bold))
                                .foregroundColor(.white)
                        }
                    }
                }
            }
            .frame(height: 200)

            // Inline legend replacing missing FlowLegend
            VStack(alignment: .leading, spacing: 8) {
                ForEach(data.operations) { op in
                    HStack(spacing: 8) {
                        Circle()
                            .fill(opColor(op.operation))
                            .frame(width: 10, height: 10)
                        Text(op.operation)
                            .font(.system(size: 12))
                            .foregroundColor(AppTheme.textPrimary)
                        Spacer()
                        Text("\(Int(op.percentage))%")
                            .font(.system(size: 12))
                            .foregroundColor(AppTheme.textMuted)
                    }
                }
            }
        }
        .glassCard()
        .padding(.horizontal, 20)

        // Horizontal bar
        VStack(alignment: .leading, spacing: 12) {
            Text("Volume by Type")
                .font(.system(size: 15, weight: .bold))
                .foregroundColor(AppTheme.textPrimary)

            Chart {
                ForEach(data.operations) { op in
                    BarMark(
                        x: .value("Amount", op.amount),
                        y: .value("Type", op.operation)
                    )
                    .foregroundStyle(opColor(op.operation))
                    .annotation(position: .trailing, alignment: .leading) {
                        Text(shortTenge(op.amount))
                            .font(.system(size: 10))
                            .foregroundColor(AppTheme.textMuted)
                    }
                }
            }
            .chartXAxis(.hidden)
            .frame(height: CGFloat(data.operations.count) * 38)
        }
        .glassCard()
        .padding(.horizontal, 20)

        // Detail list
        VStack(alignment: .leading, spacing: 8) {
            Text("Details")
                .font(.system(size: 15, weight: .bold))
                .foregroundColor(AppTheme.textPrimary)

            ForEach(data.operations) { op in
                HStack {
                    Circle()
                        .fill(opColor(op.operation))
                        .frame(width: 10, height: 10)

                    VStack(alignment: .leading, spacing: 2) {
                        Text(op.operation)
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundColor(AppTheme.textPrimary)
                        Text("\(op.count) transactions")
                            .font(.system(size: 11))
                            .foregroundColor(AppTheme.textMuted)
                    }

                    Spacer()

                    VStack(alignment: .trailing, spacing: 2) {
                        Text(formatTenge(op.amount))
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundColor(AppTheme.textPrimary)
                        Text("\(Int(op.percentage))%")
                            .font(.system(size: 11))
                            .foregroundColor(AppTheme.textMuted)
                    }
                }
                .padding(.vertical, 4)

                if op.id != data.operations.last?.id {
                    Divider()
                }
            }
        }
        .glassCard()
        .padding(.horizontal, 20)

        // Most common type insight
        if let top = data.operations.max(by: { $0.count < $1.count }) {
            HStack(spacing: 6) {
                Image(systemName: "star.fill")
                    .font(.system(size: 12))
                    .foregroundColor(AppTheme.yellow)
                Text("Most common: **\(top.operation)** (\(top.count) transactions)")
                    .font(.system(size: 13))
                    .foregroundColor(AppTheme.textPrimary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .glassCard(padding: 14)
            .padding(.horizontal, 20)
        }
    }

    // MARK: - Local formatting helpers (fallbacks if global helpers are missing)
    private func formatTenge(_ amount: Double) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.currencyCode = "KZT"
        formatter.maximumFractionDigits = 0
        return formatter.string(from: NSNumber(value: amount)) ?? ""
    }

    private func shortTenge(_ amount: Double) -> String {
        // Short scale: K, M, B for readability in annotations
        let absVal = abs(amount)
        let sign = amount < 0 ? "-" : ""
        let value: Double
        let suffix: String
        switch absVal {
        case 1_000_000_000...:
            value = absVal / 1_000_000_000
            suffix = "B"
        case 1_000_000...:
            value = absVal / 1_000_000
            suffix = "M"
        case 1_000...:
            value = absVal / 1_000
            suffix = "K"
        default:
            value = absVal
            suffix = ""
        }
        let formatted = String(format: value.truncatingRemainder(dividingBy: 1) == 0 ? "%.0f" : "%.1f", value)
        return "\(sign)\(formatted)\(suffix) ₸"
    }
}
