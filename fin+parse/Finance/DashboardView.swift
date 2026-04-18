// DashboardView.swift — with Localization + Dark/Light theme

import SwiftUI
import Charts

struct DashboardView: View {
    @EnvironmentObject var dataManager: SharedDataManager
    @EnvironmentObject var authManager: AuthManager
    @EnvironmentObject var themeManager: ThemeManager
    @EnvironmentObject var localization: LocalizationManager
    @Environment(\.colorScheme) var scheme
    @State private var chartPeriod  = 0
    @State private var showAccount  = false

    private var t: ThemedColors { ThemedColors(isDark: scheme == .dark) }
    private func s(_ key: LocalizedKey) -> String { localization.str(key) }

    private var greeting: String {
        let h = Calendar.current.component(.hour, from: Date())
        if h < 12 { return s(.goodMorning) }
        else if h < 17 { return s(.goodAfternoon) }
        else { return s(.goodEvening) }
    }

    var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 22) {
                // Top bar
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(greeting)
                            .font(.system(size: 13, weight: .medium))
                            .foregroundColor(t.textMuted)
                        Text(authManager.currentUser?.firstName ?? "there")
                            .font(.system(size: 26, weight: .black, design: .rounded))
                            .foregroundColor(t.textPrimary)
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

                // Balance Hero Card
                BalanceHeroCard()

                // Stats Row
                HStack(spacing: 12) {
                    MiniStatCard(title: s(.income),   value: dataManager.totalIncome,    color: AppTheme.green,  icon: "arrow.down.circle.fill")
                    MiniStatCard(title: s(.expenses), value: dataManager.totalExpenses,  color: AppTheme.red,    icon: "arrow.up.circle.fill")
                    MiniStatCard(title: s(.savings),  value: dataManager.savingsRate,    color: AppTheme.accent, icon: "percent", isPercent: true)
                }
                .padding(.horizontal, 20)

                // Cash Flow Chart
                VStack(alignment: .leading, spacing: 16) {
                    HStack {
                        VStack(alignment: .leading, spacing: 3) {
                            Text(s(.cashFlow)).font(.system(size: 17, weight: .bold)).foregroundColor(t.textPrimary)
                            Text(s(.incomeVsExpenses)).font(.system(size: 12)).foregroundColor(t.textMuted)
                        }
                        Spacer()
                        HStack(spacing: 4) {
                            ForEach(["6M","1Y"].indices, id: \.self) { i in
                                Button(action: { withAnimation { chartPeriod = i } }) {
                                    Text(["6M","1Y"][i])
                                        .font(.system(size: 11, weight: .semibold))
                                        .foregroundColor(chartPeriod == i ? .white : t.textMuted)
                                        .padding(.horizontal, 10).padding(.vertical, 5)
                                        .background(chartPeriod == i ? AppTheme.accent : t.surface2)
                                        .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
                                }
                                .buttonStyle(PressEffect())
                            }
                        }
                    }
                    CashFlowChart(data: dataManager.monthlyData)
                }
                .glassCard()
                .padding(.horizontal, 20)

                // Category Donut + Legend
                VStack(alignment: .leading, spacing: 16) {
                    Text(s(.spendingBreakdown))
                        .font(.system(size: 17, weight: .bold))
                        .foregroundColor(t.textPrimary)

                    if dataManager.totalExpenses > 0 {
                        HStack(spacing: 20) {
                            Chart(dataManager.categorySpending) { item in
                                SectorMark(
                                    angle: .value("Amount", item.amount),
                                    innerRadius: .ratio(0.62),
                                    angularInset: 2
                                )
                                .foregroundStyle(item.category.color)
                                .cornerRadius(4)
                            }
                            .frame(width: 130, height: 130)
                            .overlay {
                                VStack(spacing: 2) {
                                    Text("Total").font(.system(size: 10)).foregroundColor(t.textMuted)
                                    Text("₸\(String(format: "%.0f", dataManager.totalExpenses))")
                                        .font(.system(size: 15, weight: .bold)).foregroundColor(t.textPrimary)
                                }
                            }
                            VStack(alignment: .leading, spacing: 8) {
                                ForEach(dataManager.categorySpending.prefix(5)) { item in
                                    HStack(spacing: 8) {
                                        Circle().fill(item.category.color).frame(width: 8, height: 8)
                                        Text(item.category.rawValue)
                                            .font(.system(size: 12, weight: .medium))
                                            .foregroundColor(t.textPrimary)
                                        Spacer()
                                        Text("\(item.progressPercent)%")
                                            .font(.system(size: 11, weight: .semibold))
                                            .foregroundColor(t.textMuted)
                                    }
                                }
                            }
                        }
                    } else {
                        Text(s(.noExpensesYet))
                            .font(.system(size: 14))
                            .foregroundColor(t.textMuted)
                            .frame(maxWidth: .infinity, alignment: .center)
                            .padding(.vertical, 30)
                    }
                }
                .glassCard()
                .padding(.horizontal, 20)

                // Recent Transactions
                VStack(alignment: .leading, spacing: 14) {
                    HStack {
                        Text(s(.recentTransactions))
                            .font(.system(size: 17, weight: .bold))
                            .foregroundColor(t.textPrimary)
                        Spacer()
                        Text(s(.seeAll)).font(.system(size: 13, weight: .medium)).foregroundColor(AppTheme.accent)
                    }
                    ForEach(dataManager.transactions.prefix(4)) { txn in
                        TransactionRow(transaction: txn)
                    }
                }
                .glassCard()
                .padding(.horizontal, 20)
                .padding(.bottom, 100)
            }
        }
        .background(t.bg)
        .sheet(isPresented: $showAccount) {
            AccountView()
                .environmentObject(authManager)
                .environmentObject(dataManager)
                .environmentObject(themeManager)
                .environmentObject(localization)
        }
    }
}

// MARK: - Balance Hero Card
struct BalanceHeroCard: View {
    @EnvironmentObject var dataManager: SharedDataManager
    @EnvironmentObject var localization: LocalizationManager
    @Environment(\.colorScheme) var scheme
    private var t: ThemedColors { ThemedColors(isDark: scheme == .dark) }

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: AppTheme.radiusXL, style: .continuous)
                .fill(AppTheme.accentGradient)
            Circle().fill(.white.opacity(0.06)).frame(width: 200, height: 200).offset(x: 80, y: -60)
            Circle().fill(.white.opacity(0.04)).frame(width: 150, height: 150).offset(x: -60, y: 60)

            VStack(alignment: .leading, spacing: 18) {
                HStack {
                    Text(localization.str(.totalBalance))
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundColor(.white.opacity(0.7))
                    Spacer()
                    Image(systemName: "eye.fill").font(.system(size: 14)).foregroundColor(.white.opacity(0.6))
                }
                Text("₸\(String(format: "%.2f", dataManager.balance))")
                    .font(.system(size: 38, weight: .black, design: .rounded))
                    .foregroundColor(.white)

                HStack(spacing: 0) {
                    Divider().frame(width: 1, height: 30).background(.white.opacity(0.3)).padding(.horizontal, 16)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(localization.str(.thisMonth)).font(.system(size: 11)).foregroundColor(.white.opacity(0.65))
                        HStack(spacing: 4) {
                            Image(systemName: dataManager.monthlyTrend == .up ? "arrow.up.right" : "arrow.down.right")
                                .font(.system(size: 11, weight: .bold))
                            Text("₸\(String(format: "%.0f", dataManager.thisMonthExpenses)) spent")
                                .font(.system(size: 13, weight: .semibold))
                        }
                        .foregroundColor(dataManager.monthlyTrend == .up ? AppTheme.red : AppTheme.green)
                    }
                    Spacer()
                }
            }
            .padding(24)
        }
        .frame(height: 170)
        .padding(.horizontal, 20)
        .shadow(color: AppTheme.accent.opacity(0.35), radius: 24, y: 10)
    }
}

// MARK: - Mini Stat Card
struct MiniStatCard: View {
    let title: String
    let value: Double
    let color: Color
    let icon: String
    var isPercent: Bool = false
    @Environment(\.colorScheme) var scheme
    private var t: ThemedColors { ThemedColors(isDark: scheme == .dark) }

    var displayValue: String {
        isPercent ? "\(String(format: "%.1f", value))%" : "₸\(String(format: "%.0f", value))"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ZStack {
                Circle().fill(color.opacity(0.15)).frame(width: 34, height: 34)
                Image(systemName: icon).font(.system(size: 14)).foregroundColor(color)
            }
            Text(displayValue)
                .font(.system(size: 17, weight: .bold, design: .rounded))
                .foregroundColor(t.textPrimary)
            Text(title)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(t.textMuted)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(t.surface)
        .clipShape(RoundedRectangle(cornerRadius: AppTheme.radiusMD, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: AppTheme.radiusMD, style: .continuous).stroke(t.border, lineWidth: 1))
    }
}

// MARK: - Cash Flow Chart
struct CashFlowChart: View {
    let data: [MonthlyData]
    @Environment(\.colorScheme) var scheme
    private var t: ThemedColors { ThemedColors(isDark: scheme == .dark) }

    var body: some View {
        Chart {
            ForEach(data) { d in
                LineMark(x: .value("Month", d.month), y: .value("Income", d.income))
                    .foregroundStyle(AppTheme.green).interpolationMethod(.catmullRom)
                    .lineStyle(StrokeStyle(lineWidth: 2.5))
                AreaMark(x: .value("Month", d.month), y: .value("Income", d.income))
                    .foregroundStyle(AppTheme.green.opacity(0.08)).interpolationMethod(.catmullRom)
                LineMark(x: .value("Month", d.month), y: .value("Expenses", d.expenses))
                    .foregroundStyle(AppTheme.accent).interpolationMethod(.catmullRom)
                    .lineStyle(StrokeStyle(lineWidth: 2.5))
                AreaMark(x: .value("Month", d.month), y: .value("Expenses", d.expenses))
                    .foregroundStyle(AppTheme.accent.opacity(0.06)).interpolationMethod(.catmullRom)
            }
        }
        .chartXAxis {
            AxisMarks { AxisValueLabel().foregroundStyle(t.textMuted).font(.system(size: 10)) }
        }
        .chartYAxis {
            AxisMarks { AxisValueLabel().foregroundStyle(t.textMuted).font(.system(size: 10)) }
        }
        .frame(height: 160)
    }
}

// MARK: - Transaction Row
struct TransactionRow: View {
    let transaction: Transaction
    @Environment(\.colorScheme) var scheme
    private var t: ThemedColors { ThemedColors(isDark: scheme == .dark) }

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 11, style: .continuous)
                    .fill(transaction.category.softColor)
                    .frame(width: 42, height: 42)
                Image(systemName: transaction.category.icon)
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(transaction.category.color)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(transaction.title)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(t.textPrimary)
                Text(transaction.shortDate)
                    .font(.system(size: 11))
                    .foregroundColor(t.textMuted)
            }
            Spacer()
            Text(transaction.formattedAmount)
                .font(.system(size: 15, weight: .bold, design: .rounded))
                .foregroundColor(transaction.isExpense ? t.textPrimary : AppTheme.green)
        }
        .padding(.vertical, 4)
    }
}

extension CategorySpending {
    var progressPercent: Int { Int(percentage * 100) }
}
