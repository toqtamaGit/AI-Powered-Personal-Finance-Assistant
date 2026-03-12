// ContentView.swift — Main tab container

import SwiftUI
import Swift

struct ContentView: View {
    @State private var selectedTab = 0

    var body: some View {
        ZStack(alignment: .bottom) {
            AppTheme.bg.ignoresSafeArea()

            // Pages
            ZStack {
                DashboardView().opacity(selectedTab == 0 ? 1 : 0)
                TransactionsView().opacity(selectedTab == 1 ? 1 : 0)
                BudgetView().opacity(selectedTab == 2 ? 1 : 0)
                GoalsView().opacity(selectedTab == 3 ? 1 : 0)
                ChatView().opacity(selectedTab == 4 ? 1 : 0)
            }

            FincoraTabBar(selected: $selectedTab)
        }
        .ignoresSafeArea(edges: .bottom)
    }
}

// MARK: - Custom Tab Bar
struct FincoraTabBar: View {
    @Binding var selected: Int
    @Namespace private var ns

    struct Tab {
        let icon: String; let active: String; let label: String
    }
    let tabs: [Tab] = [
        Tab(icon: "chart.pie",       active: "chart.pie.fill",       label: "Dashboard"),
        Tab(icon: "arrow.up.arrow.down", active: "arrow.up.arrow.down", label: "Transactions"),
        Tab(icon: "dollarsign.circle", active: "dollarsign.circle.fill", label: "Budget"),
        Tab(icon: "target",           active: "target",               label: "Goals"),
        Tab(icon: "bubble.left",      active: "bubble.left.fill",     label: "Chat"),
    ]

    var body: some View {
        HStack(spacing: 0) {
            ForEach(tabs.indices, id: \.self) { i in
                Button(action: {
                    withAnimation(.spring(response: 0.35, dampingFraction: 0.75)) { selected = i }
                }) {
                    VStack(spacing: 5) {
                        ZStack {
                            if selected == i {
                                RoundedRectangle(cornerRadius: 12, style: .continuous)
                                    .fill(AppTheme.accentSoft)
                                    .frame(width: 46, height: 34)
                                    .matchedGeometryEffect(id: "TabBG", in: ns)
                            }
                            Image(systemName: selected == i ? tabs[i].active : tabs[i].icon)
                                .font(.system(size: 18, weight: selected == i ? .semibold : .regular))
                                .foregroundColor(selected == i ? AppTheme.accent : AppTheme.textMuted)
                                .symbolEffect(.bounce, value: selected == i)
                        }
                        .frame(width: 46, height: 34)

                        Text(tabs[i].label)
                            .font(.system(size: 9.5, weight: selected == i ? .semibold : .regular))
                            .foregroundColor(selected == i ? AppTheme.accent : AppTheme.textMuted)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.top, 10)
                    .padding(.bottom, 20)
                }
                .buttonStyle(PressEffect())
            }
        }
        .background(
            AppTheme.surface
                .overlay(Rectangle().frame(height: 1).foregroundColor(AppTheme.border), alignment: .top)
        )
    }
}

#Preview {
    ContentView()
        .environmentObject(SharedDataManager())
        .environmentObject(AuthManager())
}
