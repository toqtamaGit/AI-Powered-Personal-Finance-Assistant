// AccountView.swift — Profile & Settings Sheet (with Language + Dark Mode)

import SwiftUI

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Account Sheet (root)
// ─────────────────────────────────────────────────────────────────────────────
struct AccountView: View {
    @EnvironmentObject var authManager: AuthManager
    @EnvironmentObject var dataManager: SharedDataManager
    @EnvironmentObject var themeManager: ThemeManager
    @EnvironmentObject var localization: LocalizationManager
    @Environment(\.dismiss) private var dismiss
    @Environment(\.colorScheme) var scheme

    // Edit-profile
    @State private var editingName  = false
    @State private var draftName    = ""

    // Settings toggles
    @State private var notificationsOn  = true
    @State private var budgetAlertsOn   = true
    @State private var weeklyReportOn   = false

    // Danger zone
    @State private var showLogoutAlert  = false
    @State private var showDeleteAlert  = false

    private var t: ThemedColors { ThemedColors(isDark: scheme == .dark) }
    private func s(_ key: LocalizedKey) -> String { localization.str(key) }

    var body: some View {
        NavigationView {
            ZStack {
                t.bg.ignoresSafeArea()

                ScrollView(showsIndicators: false) {
                    VStack(spacing: 0) {
                        profileHeader
                        statsRow.padding(.horizontal, 20).padding(.top, 24)
                        appearanceSection.padding(.top, 24)
                        languageSection
                        settingsSections
                        dangerZone
                        Spacer().frame(height: 60)
                    }
                }
            }
            .navigationTitle(s(.account))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button { dismiss() } label: {
                        Image(systemName: "xmark")
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundColor(t.textMuted)
                            .padding(8)
                            .background(t.surface2)
                            .clipShape(Circle())
                    }
                }
            }
            .alert(s(.editName), isPresented: $editingName) {
                TextField(s(.fullName), text: $draftName)
                    .textInputAutocapitalization(.words)
                Button(s(.save)) { saveName() }
                Button(s(.cancel), role: .cancel) { }
            } message: { Text(s(.enterNewName)) }
            .alert(s(.signOutConfirm), isPresented: $showLogoutAlert) {
                Button(s(.signOut), role: .destructive) { authManager.logout() }
                Button(s(.cancel), role: .cancel) { }
            } message: { Text(s(.signOutMessage)) }
            .alert(s(.deleteAccountConfirm), isPresented: $showDeleteAlert) {
                Button(s(.delete), role: .destructive) { authManager.logout() }
                Button(s(.cancel), role: .cancel) { }
            } message: { Text(s(.deleteAccountMessage)) }
        }
    }

    // ─────────────────────────────────────────────
    // MARK: Profile header
    // ─────────────────────────────────────────────
    private var profileHeader: some View {
        VStack(spacing: 0) {
            ZStack(alignment: .bottom) {
                AppTheme.accentGradient
                    .frame(height: 120)
                    .ignoresSafeArea(edges: .top)
                Circle().fill(.white.opacity(0.06)).frame(width: 160, height: 160).offset(x: 100, y: -30)
                Circle().fill(.white.opacity(0.04)).frame(width: 100, height: 100).offset(x: -80, y: 20)
                ZStack {
                    Circle()
                        .fill(AppTheme.accentGradient)
                        .frame(width: 80, height: 80)
                        .overlay(Circle().stroke(.white, lineWidth: 4))
                        .shadow(color: AppTheme.accent.opacity(0.4), radius: 12, y: 6)
                    Text(authManager.currentUser?.initials ?? "?")
                        .font(.system(size: 28, weight: .bold))
                        .foregroundColor(.white)
                }
                .offset(y: 40)
            }

            VStack(spacing: 6) {
                Button(action: {
                    draftName = authManager.currentUser?.fullName ?? ""
                    editingName = true
                }) {
                    HStack(spacing: 6) {
                        Text(authManager.currentUser?.fullName ?? "—")
                            .font(.system(size: 20, weight: .bold))
                            .foregroundColor(t.textPrimary)
                        Image(systemName: "pencil.circle.fill")
                            .font(.system(size: 16))
                            .foregroundColor(AppTheme.accent)
                    }
                }
                .buttonStyle(PressEffect())

                Text(authManager.currentUser?.email ?? "")
                    .font(.system(size: 13))
                    .foregroundColor(t.textMuted)
            }
            .padding(.top, 50)
            .padding(.bottom, 20)
        }
    }

    // ─────────────────────────────────────────────
    // MARK: Stats row
    // ─────────────────────────────────────────────
    private var statsRow: some View {
        HStack(spacing: 12) {
            statChip(
                label: s(.memberSince),
                value: formattedMemberSince,
                icon: "calendar",
                color: AppTheme.accent
            )
            statChip(
                label: s(.totalTransactions),
                value: "\(dataManager.transactions.count)",
                icon: "arrow.left.arrow.right",
                color: AppTheme.green
            )
        }
    }

    private func statChip(label: String, value: String, icon: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ZStack {
                Circle().fill(color.opacity(0.15)).frame(width: 34, height: 34)
                Image(systemName: icon).font(.system(size: 14)).foregroundColor(color)
            }
            Text(value).font(.system(size: 17, weight: .bold, design: .rounded)).foregroundColor(t.textPrimary)
            Text(label).font(.system(size: 11)).foregroundColor(t.textMuted)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(t.surface)
        .clipShape(RoundedRectangle(cornerRadius: AppTheme.radiusMD, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: AppTheme.radiusMD).stroke(t.border))
    }

    private var formattedMemberSince: String {
        guard let date = authManager.currentUser?.createdAt else { return "—" }
        let f = DateFormatter(); f.dateFormat = "MMM yyyy"
        return f.string(from: date)
    }

    // ─────────────────────────────────────────────
    // MARK: Appearance Section (Dark Mode)
    // ─────────────────────────────────────────────
    private var appearanceSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader(s(.appearance), icon: "paintbrush.fill")

            VStack(spacing: 0) {
                // Dark Mode Toggle
                HStack(spacing: 14) {
                    settingIcon(systemName: themeManager.isDarkMode ? "moon.fill" : "sun.max.fill",
                                color: themeManager.isDarkMode ? AppTheme.purple : AppTheme.yellow)
                    Text(s(.darkMode))
                        .font(.system(size: 15, weight: .medium))
                        .foregroundColor(t.textPrimary)
                    Spacer()
                    Toggle("", isOn: $themeManager.isDarkMode)
                        .tint(AppTheme.accent)
                        .labelsHidden()
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 14)

                // Segmented mode picker
                VStack(alignment: .leading, spacing: 10) {
                    Text(s(.appearance))
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(t.textMuted)
                        .padding(.horizontal, 16)

                    HStack(spacing: 8) {
                        ForEach([("sun.max.fill", false, "Light"), ("moon.fill", true, "Dark")], id: \.2) { icon, dark, label in
                            Button(action: { withAnimation(.spring(response: 0.3)) { themeManager.isDarkMode = dark } }) {
                                VStack(spacing: 8) {
                                    ZStack {
                                        RoundedRectangle(cornerRadius: AppTheme.radiusSM)
                                            .fill(dark ? Color(hex: "#1C1C26") : Color(hex: "#F9FAFB"))
                                            .frame(height: 56)
                                            .overlay(
                                                RoundedRectangle(cornerRadius: AppTheme.radiusSM)
                                                    .stroke(themeManager.isDarkMode == dark ? AppTheme.accent : t.border, lineWidth: themeManager.isDarkMode == dark ? 2 : 1)
                                            )
                                        Image(systemName: icon)
                                            .font(.system(size: 20))
                                            .foregroundColor(dark ? Color(hex: "#F1F1F5") : Color(hex: "#111827"))
                                    }
                                    Text(label)
                                        .font(.system(size: 11, weight: themeManager.isDarkMode == dark ? .semibold : .regular))
                                        .foregroundColor(themeManager.isDarkMode == dark ? AppTheme.accent : t.textMuted)
                                }
                            }
                            .buttonStyle(PressEffect())
                            .frame(maxWidth: .infinity)
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.bottom, 14)
                }
            }
            .background(t.surface)
            .clipShape(RoundedRectangle(cornerRadius: AppTheme.radiusLG, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: AppTheme.radiusLG).stroke(t.border))
        }
        .padding(.horizontal, 20)
    }

    // ─────────────────────────────────────────────
    // MARK: Language Section
    // ─────────────────────────────────────────────
    private var languageSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader(s(.language), icon: "globe")

            VStack(spacing: 0) {
                ForEach(Array(AppLanguage.allCases.enumerated()), id: \.element) { index, lang in
                    Button(action: {
                        withAnimation(.spring(response: 0.3)) {
                            localization.language = lang
                        }
                    }) {
                        HStack(spacing: 14) {
                            Text(lang.flag)
                                .font(.system(size: 22))
                                .frame(width: 36, height: 36)
                                .background(t.surface2)
                                .clipShape(RoundedRectangle(cornerRadius: 10))

                            Text(lang.displayName)
                                .font(.system(size: 15, weight: .medium))
                                .foregroundColor(t.textPrimary)

                            Spacer()

                            if localization.language == lang {
                                Image(systemName: "checkmark.circle.fill")
                                    .font(.system(size: 20))
                                    .foregroundColor(AppTheme.accent)
                                    .transition(.scale.combined(with: .opacity))
                            } else {
                                Circle()
                                    .stroke(t.border, lineWidth: 1.5)
                                    .frame(width: 20, height: 20)
                            }
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 14)
                        .background(localization.language == lang ? AppTheme.accentSoft : Color.clear)
                    }
                    .buttonStyle(PressEffect())

                    if index < AppLanguage.allCases.count - 1 {
                        Divider()
                            .padding(.leading, 66)
                            .foregroundColor(t.border)
                    }
                }
            }
            .background(t.surface)
            .clipShape(RoundedRectangle(cornerRadius: AppTheme.radiusLG, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: AppTheme.radiusLG).stroke(t.border))
        }
        .padding(.horizontal, 20)
        .padding(.top, 24)
    }

    // ─────────────────────────────────────────────
    // MARK: Settings sections (notifications)
    // ─────────────────────────────────────────────
    private var settingsSections: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader(s(.notifications), icon: "bell.fill")

            VStack(spacing: 0) {
                toggleRow(label: s(.budgetAlerts),  icon: "exclamationmark.triangle.fill", color: AppTheme.yellow,  binding: $budgetAlertsOn)
                Divider().padding(.leading, 66).foregroundColor(t.border)
                toggleRow(label: s(.weeklyReport),  icon: "chart.bar.fill",               color: AppTheme.accent,  binding: $weeklyReportOn)
            }
            .background(t.surface)
            .clipShape(RoundedRectangle(cornerRadius: AppTheme.radiusLG, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: AppTheme.radiusLG).stroke(t.border))

            sectionHeader(s(.support), icon: "questionmark.circle.fill").padding(.top, 12)

            VStack(spacing: 0) {
                navRow(label: s(.helpCenter),    icon: "book.fill",         color: AppTheme.teal)
                Divider().padding(.leading, 66).foregroundColor(t.border)
                navRow(label: s(.privacyPolicy), icon: "lock.shield.fill",  color: AppTheme.green)
                Divider().padding(.leading, 66).foregroundColor(t.border)
                HStack(spacing: 14) {
                    settingIcon(systemName: "info.circle.fill", color: t.textMuted)
                    Text(s(.version))
                        .font(.system(size: 15, weight: .medium))
                        .foregroundColor(t.textPrimary)
                    Spacer()
                    Text("1.0.0")
                        .font(.system(size: 14))
                        .foregroundColor(t.textMuted)
                }
                .padding(.horizontal, 16).padding(.vertical, 14)
            }
            .background(t.surface)
            .clipShape(RoundedRectangle(cornerRadius: AppTheme.radiusLG, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: AppTheme.radiusLG).stroke(t.border))
        }
        .padding(.horizontal, 20)
        .padding(.top, 24)
    }

    private var dangerZone: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader(s(.dangerZone), icon: "exclamationmark.triangle.fill", color: AppTheme.red)

            VStack(spacing: 0) {
                Button(action: { showLogoutAlert = true }) {
                    HStack(spacing: 14) {
                        settingIcon(systemName: "arrow.right.square.fill", color: AppTheme.red)
                        Text(s(.signOut))
                            .font(.system(size: 15, weight: .medium))
                            .foregroundColor(AppTheme.red)
                        Spacer()
                        Image(systemName: "chevron.right")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundColor(t.textMuted)
                    }
                    .padding(.horizontal, 16).padding(.vertical, 14)
                }
                .buttonStyle(PressEffect())

                Divider().padding(.leading, 66).foregroundColor(t.border)

                Button(action: { showDeleteAlert = true }) {
                    HStack(spacing: 14) {
                        settingIcon(systemName: "trash.fill", color: AppTheme.red)
                        Text(s(.deleteAccount))
                            .font(.system(size: 15, weight: .medium))
                            .foregroundColor(AppTheme.red)
                        Spacer()
                        Image(systemName: "chevron.right")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundColor(t.textMuted)
                    }
                    .padding(.horizontal, 16).padding(.vertical, 14)
                }
                .buttonStyle(PressEffect())
            }
            .background(AppTheme.redSoft)
            .clipShape(RoundedRectangle(cornerRadius: AppTheme.radiusLG, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: AppTheme.radiusLG).stroke(AppTheme.red.opacity(0.2)))
        }
        .padding(.horizontal, 20)
        .padding(.top, 24)
    }

    // ─────────────────────────────────────────────
    // MARK: Reusable row components
    // ─────────────────────────────────────────────
    private func sectionHeader(_ title: String, icon: String, color: Color = AppTheme.textMuted) -> some View {
        Label(title, systemImage: icon)
            .font(.system(size: 12, weight: .semibold))
            .foregroundColor(color)
            .textCase(.uppercase)
    }

    private func settingIcon(systemName: String, color: Color) -> some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(color.opacity(0.15))
                .frame(width: 34, height: 34)
            Image(systemName: systemName)
                .font(.system(size: 15))
                .foregroundColor(color)
        }
    }

    private func toggleRow(label: String, icon: String, color: Color, binding: Binding<Bool>) -> some View {
        HStack(spacing: 14) {
            settingIcon(systemName: icon, color: color)
            Text(label)
                .font(.system(size: 15, weight: .medium))
                .foregroundColor(t.textPrimary)
            Spacer()
            Toggle("", isOn: binding)
                .tint(AppTheme.accent)
                .labelsHidden()
        }
        .padding(.horizontal, 16).padding(.vertical, 14)
    }

    private func navRow(label: String, icon: String, color: Color) -> some View {
        HStack(spacing: 14) {
            settingIcon(systemName: icon, color: color)
            Text(label)
                .font(.system(size: 15, weight: .medium))
                .foregroundColor(t.textPrimary)
            Spacer()
            Image(systemName: "chevron.right")
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(t.textMuted)
        }
        .padding(.horizontal, 16).padding(.vertical, 14)
    }

    // ─────────────────────────────────────────────
    // MARK: Actions
    // ─────────────────────────────────────────────
    private func saveName() {
        guard !draftName.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        // Update user name via AuthManager (would need extending AuthManager)
        // For now we log it
        print("Saving name: \(draftName)")
    }
}
