// DashboardView.swift — Main analytics hub with four sub-pages.

import SwiftUI
import Charts

// MARK: - Analytics Tab Pages
enum AnalyticsPage: String, CaseIterable, Identifiable {
    case overview   = "Overview"
    case categories = "Categories"
    case banks      = "Banks"
    case types      = "Types"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .overview:   return "chart.pie"
        case .categories: return "tag"
        case .banks:      return "building.columns"
        case .types:      return "arrow.left.arrow.right"
        }
    }
}

// MARK: - Main Dashboard View
struct DashboardView: View {
    @EnvironmentObject var authManager: AuthManager
    @StateObject private var service = DashboardService()
    @State private var page: AnalyticsPage = .overview
    @State private var showAccount = false

    private var greeting: String {
        let h = Calendar.current.component(.hour, from: Date())
        if h < 12 { return "Good morning" }
        else if h < 17 { return "Good afternoon" }
        else { return "Good evening" }
    }

    var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 18) {
                // Top bar
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(greeting)
                            .font(.system(size: 13, weight: .medium))
                            .foregroundColor(AppTheme.textMuted)
                        Text(authManager.currentUser?.firstName ?? "there")
                            .font(.system(size: 26, weight: .black, design: .rounded))
                            .foregroundColor(AppTheme.textPrimary)
                    }
                    Spacer()
                    Button { showAccount = true } label: {
                        ZStack {
                            Circle()
                                .fill(AppTheme.accentGradient)
                                .frame(width: 42, height: 42)
                                .shadow(color: AppTheme.accent.opacity(0.3), radius: 8, y: 3)
                            Text(authManager.currentUser?.initials ?? "?")
                                .font(.system(size: 15, weight: .bold))
                                .foregroundColor(.white)
                        }
                    }
                    .buttonStyle(PressEffect())
                }
                .padding(.horizontal, 20)
                .padding(.top, 16)

                // Page picker
                HStack(spacing: 4) {
                    ForEach(AnalyticsPage.allCases) { p in
                        Button(action: { withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) { page = p } }) {
                            HStack(spacing: 4) {
                                Image(systemName: p.icon)
                                    .font(.system(size: 11))
                                Text(p.rawValue)
                                    .font(.system(size: 12, weight: .semibold))
                            }
                            .foregroundColor(page == p ? .white : AppTheme.textMuted)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 7)
                            .background(page == p ? AppTheme.accent : AppTheme.surface2)
                            .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
                        }
                        .buttonStyle(PressEffect())
                    }
                }
                .padding(.horizontal, 20)

                // Shared filter bar
                FilterBar(service: service)

                // Loading / error
                if service.isLoading {
                    ProgressView()
                        .frame(maxWidth: .infinity, minHeight: 200)
                } else if let err = service.errorMessage {
                    VStack(spacing: 8) {
                        Image(systemName: "wifi.exclamationmark")
                            .font(.system(size: 28))
                            .foregroundColor(AppTheme.red)
                        Text(err)
                            .font(.system(size: 13))
                            .foregroundColor(AppTheme.textMuted)
                            .multilineTextAlignment(.center)
                        Button("Retry") { Task { await service.fetchAll() } }
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundColor(AppTheme.accent)
                    }
                    .frame(maxWidth: .infinity, minHeight: 200)
                    .padding()
                } else {
                    // Page content
                    switch page {
                    case .overview:   OverviewPage(service: service)
                    case .categories: CategoryAnalysisPage(service: service)
                    case .banks:      BankAnalysisPage(service: service)
                    case .types:      TransactionTypePage(service: service)
                    }
                }

                Spacer(minLength: 40)
            }
        }
        .background(AppTheme.bg.ignoresSafeArea())
        .sheet(isPresented: $showAccount) { AccountView().environmentObject(authManager) }
        .task { await service.fetchAll() }
    }
}

// MARK: - Overview Page
struct OverviewPage: View {
    @ObservedObject var service: DashboardService

    var body: some View {
        VStack(spacing: 16) {
            // KPI cards
            if let ov = service.overview {
                // Balance hero
                VStack(spacing: 6) {
                    Text("Balance")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(AppTheme.textMuted)
                    Text(formatTenge(ov.balance))
                        .font(.system(size: 32, weight: .black, design: .rounded))
                        .foregroundColor(ov.balance >= 0 ? AppTheme.green : AppTheme.red)
                    Text("\(ov.transactionCount) transactions")
                        .font(.system(size: 12))
                        .foregroundColor(AppTheme.textMuted)
                }
                .frame(maxWidth: .infinity)
                .glassCard()
                .padding(.horizontal, 20)

                // Income / Expenses row
                HStack(spacing: 12) {
                    MiniKPI(title: "Income", value: ov.totalIncome, color: AppTheme.green, icon: "arrow.down.circle.fill")
                    MiniKPI(title: "Spent", value: abs(ov.totalSpent), color: AppTheme.red, icon: "arrow.up.circle.fill")
                }
                .padding(.horizontal, 20)

                // Monthly trend cards
                HStack(spacing: 12) {
                    MiniKPI(title: "This Month", value: ov.thisMonthSpent, color: AppTheme.orange, icon: "calendar")
                    MiniKPI(title: "Last Month", value: ov.lastMonthSpent, color: AppTheme.teal, icon: "calendar.badge.clock")
                }
                .padding(.horizontal, 20)

                // Trend arrow
                if ov.thisMonthSpent > 0 || ov.lastMonthSpent > 0 {
                    HStack(spacing: 6) {
                        Image(systemName: ov.trendDirection == .up ? "arrow.up.right" : "arrow.down.right")
                            .font(.system(size: 13, weight: .bold))
                            .foregroundColor(ov.trendDirection == .up ? AppTheme.red : AppTheme.green)
                        Text(ov.trendDirection == .up ? "Spending increased" : "Spending decreased")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundColor(AppTheme.textMuted)
                    }
                    .padding(.horizontal, 20)
                }

                // Top category badge
                if let top = ov.topCategory {
                    HStack(spacing: 6) {
                        Image(systemName: "star.fill")
                            .font(.system(size: 12))
                            .foregroundColor(AppTheme.yellow)
                        Text("Top category: **\(top)**")
                            .font(.system(size: 13))
                            .foregroundColor(AppTheme.textPrimary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .glassCard(padding: 14)
                    .padding(.horizontal, 20)
                }
            }

            // Category donut (mini)
            if let cats = service.categories, !cats.categories.isEmpty {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Spending by Category")
                        .font(.system(size: 15, weight: .bold))
                        .foregroundColor(AppTheme.textPrimary)

                    Chart {
                        ForEach(cats.categories) { c in
                            SectorMark(
                                angle: .value("Amount", c.amount),
                                innerRadius: .ratio(0.55),
                                angularInset: 1.5
                            )
                            .foregroundStyle(c.appCategory.color)
                        }
                    }
                    .frame(height: 180)

                    // Legend
                    FlowLegend(items: cats.categories.map { ($0.category, $0.appCategory.color, $0.percentage) })
                }
                .glassCard()
                .padding(.horizontal, 20)
            }

            // Trend line chart
            if let t = service.trend, !t.trend.isEmpty {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Monthly Trend")
                        .font(.system(size: 15, weight: .bold))
                        .foregroundColor(AppTheme.textPrimary)

                    Chart {
                        ForEach(t.trend) { m in
                            LineMark(
                                x: .value("Month", m.label),
                                y: .value("Expenses", m.expenses)
                            )
                            .foregroundStyle(AppTheme.red)
                            .symbol(Circle())

                            LineMark(
                                x: .value("Month", m.label),
                                y: .value("Income", m.income)
                            )
                            .foregroundStyle(AppTheme.green)
                            .symbol(Circle())
                        }
                    }
                    .chartYAxis {
                        AxisMarks(position: .leading) { value in
                            AxisGridLine()
                            AxisValueLabel {
                                if let v = value.as(Double.self) {
                                    Text(shortTenge(v))
                                        .font(.system(size: 9))
                                }
                            }
                        }
                    }
                    .frame(height: 180)

                    // Legend
                    HStack(spacing: 16) {
                        LegendDot(color: AppTheme.red, label: "Expenses")
                        LegendDot(color: AppTheme.green, label: "Income")
                    }
                }
                .glassCard()
                .padding(.horizontal, 20)
            }
        }
    }
}

// MARK: - Helpers

func formatTenge(_ value: Double) -> String {
    let formatter = NumberFormatter()
    formatter.numberStyle = .decimal
    formatter.maximumFractionDigits = 0
    formatter.groupingSeparator = " "
    let s = formatter.string(from: NSNumber(value: abs(value))) ?? "0"
    let sign = value < 0 ? "-" : ""
    return "\(sign)\(s) \u{20B8}"
}

func shortTenge(_ value: Double) -> String {
    if value >= 1_000_000 { return String(format: "%.1fM", value / 1_000_000) }
    if value >= 1_000 { return String(format: "%.0fK", value / 1_000) }
    return String(format: "%.0f", value)
}

struct MiniKPI: View {
    let title: String
    let value: Double
    let color: Color
    let icon: String
    var isPercent: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 12))
                    .foregroundColor(color)
                Text(title)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(AppTheme.textMuted)
            }
            Text(isPercent ? String(format: "%.0f%%", value) : formatTenge(value))
                .font(.system(size: 16, weight: .bold))
                .foregroundColor(AppTheme.textPrimary)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassCard(padding: 14)
    }
}

struct LegendDot: View {
    let color: Color
    let label: String

    var body: some View {
        HStack(spacing: 4) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(label).font(.system(size: 11)).foregroundColor(AppTheme.textMuted)
        }
    }
}

struct FlowLegend: View {
    let items: [(String, Color, Double)]

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    HStack(spacing: 5) {
                        Circle().fill(item.1).frame(width: 8, height: 8)
                        Text("\(item.0) \(String(format: "%.0f", item.2))%")
                            .font(.system(size: 11))
                            .foregroundColor(AppTheme.textMuted)
                    }
                }
            }
            Spacer()
        }
    }
}

#Preview {
    DashboardView()
        .environmentObject(AuthManager())
        .environmentObject(SharedDataManager())
}
