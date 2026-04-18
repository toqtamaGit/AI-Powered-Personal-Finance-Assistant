
import SwiftUI

@main
struct FincoraApp: App {
    @StateObject private var authManager   = AuthManager()
    @StateObject private var dataManager   = SharedDataManager()
    @StateObject private var themeManager  = ThemeManager()
    @StateObject private var localization  = LocalizationManager()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(authManager)
                .environmentObject(dataManager)
                .environmentObject(themeManager)
                .environmentObject(localization)
                .preferredColorScheme(themeManager.isDarkMode ? .dark : .light)
        }
    }
}

// MARK: - Root View
struct RootView: View {
    @EnvironmentObject var authManager: AuthManager

    var body: some View {
        if authManager.isAuthenticated {
            MainTabView()
        } else {
            AuthRootView()
        }
    }
}

// MARK: - Main Tab View
// Replace your existing ContentView / TabView with this
struct MainTabView: View {
    @EnvironmentObject var localization: LocalizationManager
    @EnvironmentObject var themeManager: ThemeManager
    @Environment(\.colorScheme) var scheme
    private var t: ThemedColors { ThemedColors(isDark: scheme == .dark) }

    var body: some View {
        TabView {
            DashboardView()
                .tabItem {
                    Label(localization.str(.totalBalance), systemImage: "house.fill")
                }

            TransactionsView()
                .tabItem {
                    Label(localization.str(.transactions), systemImage: "arrow.left.arrow.right")
                }

            BudgetView()
                .tabItem {
                    Label(localization.str(.budget), systemImage: "chart.bar.fill")
                }

            GoalsView()
                .tabItem {
                    Label(localization.str(.goals), systemImage: "target")
                }

            ChatView()
                .tabItem {
                    Label(localization.str(.aiAdvisor), systemImage: "bubble.left.and.bubble.right.fill")
                }
        }
        .tint(AppTheme.accent)
    }
}
