// AppTheme.swift — Design tokens for FinaApp (Dark/Light mode support)

import SwiftUI
import Combine
// MARK: - Theme Manager
class ThemeManager: ObservableObject {
    @Published var isDarkMode: Bool {
        didSet {
            UserDefaults.standard.set(isDarkMode, forKey: "fincora_dark_mode")
        }
    }

    init() {
        // Default to system preference if never set
        if UserDefaults.standard.object(forKey: "fincora_dark_mode") == nil {
            self.isDarkMode = false
        } else {
            self.isDarkMode = UserDefaults.standard.bool(forKey: "fincora_dark_mode")
        }
    }

    var colorScheme: ColorScheme {
        isDarkMode ? .dark : .light
    }
}

// MARK: - AppTheme (Dynamic)
struct AppTheme {
    // MARK: - Brand & Status Colors (same across modes)
    static let accent      = Color(hex: "#5D5CDE")
    static let accentSoft  = Color(hex: "#5D5CDE").opacity(0.12)
    static let green       = Color(hex: "#10B981")
    static let greenSoft   = Color(hex: "#10B981").opacity(0.15)
    static let red         = Color(hex: "#EF4444")
    static let redSoft     = Color(hex: "#EF4444").opacity(0.12)
    static let yellow      = Color(hex: "#F59E0B")
    static let yellowSoft  = Color(hex: "#F59E0B").opacity(0.12)
    static let teal        = Color(hex: "#06B6D4")
    static let teals       = Color(hex: "#06B6D4")
    static let orange      = Color(hex: "#F97316")
    static let purple      = Color(hex: "#7C3AED")

    // MARK: - Shadows
    static let shadowXS    = Color.black.opacity(0.03)
    static let shadowSM    = Color.black.opacity(0.07)
    static let shadowMD    = Color.black.opacity(0.12)

    // MARK: - Adaptive Structural Colors
    // These use SwiftUI's adaptive colors via the environment
    static var bg: Color           { Color("AppBg") }
    static var surface: Color      { Color("AppSurface") }
    static var surface2: Color     { Color("AppSurface2") }
    static var border: Color       { Color("AppBorder") }
    static var textPrimary: Color  { Color("AppTextPrimary") }
    static var textMuted: Color    { Color("AppTextMuted") }

    // MARK: - Light mode explicit colors
    static let bgLight         = Color(hex: "#F9FAFB")
    static let surfaceLight    = Color.white
    static let surface2Light   = Color(hex: "#F3F4F6")
    static let borderLight     = Color.black.opacity(0.06)
    static let textPrimaryLight = Color(hex: "#111827")
    static let textMutedLight  = Color(hex: "#6B7280")

    // MARK: - Dark mode explicit colors
    static let bgDark          = Color(hex: "#0F0F14")
    static let surfaceDark     = Color(hex: "#1C1C26")
    static let surface2Dark    = Color(hex: "#2A2A38")
    static let borderDark      = Color.white.opacity(0.08)
    static let textPrimaryDark = Color(hex: "#F1F1F5")
    static let textMutedDark   = Color(hex: "#8B8BA0")

    // MARK: - Gradients (same across modes)
    static let accentGradient = LinearGradient(
        colors: [Color(hex: "#6366F1"), Color(hex: "#8B5CF6")],
        startPoint: .topLeading, endPoint: .bottomTrailing
    )
    static let greenGradient = LinearGradient(
        colors: [Color(hex: "#10B981"), Color(hex: "#34D399")],
        startPoint: .topLeading, endPoint: .bottomTrailing
    )
    static var cardGradient: LinearGradient {
        LinearGradient(
            colors: [Color("AppSurface"), Color("AppBg")],
            startPoint: .topLeading, endPoint: .bottomTrailing
        )
    }

    // MARK: - Design Constants
    static let radiusSM: CGFloat = 10
    static let radiusMD: CGFloat = 16
    static let radiusLG: CGFloat = 22
    static let radiusXL: CGFloat = 28
}

// MARK: - Theme-aware color helper
/// Use this in views that need to respond to theme changes dynamically
struct ThemedColors {
    let isDark: Bool

    var bg: Color          { isDark ? AppTheme.bgDark         : AppTheme.bgLight }
    var surface: Color     { isDark ? AppTheme.surfaceDark     : AppTheme.surfaceLight }
    var surface2: Color    { isDark ? AppTheme.surface2Dark    : AppTheme.surface2Light }
    var border: Color      { isDark ? AppTheme.borderDark      : AppTheme.borderLight }
    var textPrimary: Color { isDark ? AppTheme.textPrimaryDark : AppTheme.textPrimaryLight }
    var textMuted: Color   { isDark ? AppTheme.textMutedDark   : AppTheme.textMutedLight }
}

// MARK: - Color hex init
extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3:
            (a, r, g, b) = (255, (int >> 8)*17, (int >> 4 & 0xF)*17, (int & 0xF)*17)
        case 6:
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8:
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (1, 1, 1, 0)
        }
        self.init(.sRGB,
            red: Double(r)/255, green: Double(g)/255,
            blue: Double(b)/255, opacity: Double(a)/255)
    }
}

// MARK: - View Modifiers
struct GlassCard: ViewModifier {
    var padding: CGFloat = 20
    @Environment(\.colorScheme) var scheme

    func body(content: Content) -> some View {
        let c = ThemedColors(isDark: scheme == .dark)
        return content
            .padding(padding)
            .background(c.surface)
            .clipShape(RoundedRectangle(cornerRadius: AppTheme.radiusLG, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: AppTheme.radiusLG, style: .continuous)
                    .stroke(c.border, lineWidth: 1)
            )
    }
}

struct PressEffect: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.97 : 1)
            .animation(.spring(response: 0.25, dampingFraction: 0.7), value: configuration.isPressed)
    }
}

extension View {
    func glassCard(padding: CGFloat = 20) -> some View {
        modifier(GlassCard(padding: padding))
    }
}

// MARK: - Color Assets fallback
// Since we can't add to Assets.xcassets directly in code, we use a workaround:
// Provide a ViewModifier that injects the right colors based on ThemeManager.
// In your actual Xcode project, add these Color Set names to Assets.xcassets:
// AppBg, AppSurface, AppSurface2, AppBorder, AppTextPrimary, AppTextMuted
// — with Light/Dark variants matching the hex values above.
//
// Alternative: Use the ThemeAwareView wrapper below in your root ContentView.

struct ThemeAwareModifier: ViewModifier {
    @ObservedObject var themeManager: ThemeManager

    func body(content: Content) -> some View {
        content
            .preferredColorScheme(themeManager.isDarkMode ? .dark : .light)
    }
}

extension View {
    func applyTheme(_ themeManager: ThemeManager) -> some View {
        modifier(ThemeAwareModifier(themeManager: themeManager))
    }
}
