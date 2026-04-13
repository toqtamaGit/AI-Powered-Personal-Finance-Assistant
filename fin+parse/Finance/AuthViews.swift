// AuthViews.swift — Login, Register, Forgot Password
// Updated: dark/light theme + localization

import SwiftUI

// MARK: - Auth Root
struct AuthRootView: View {
    @EnvironmentObject var themeManager: ThemeManager
    @EnvironmentObject var localization: LocalizationManager
    @Environment(\.colorScheme) var scheme
    @State private var selectedTab = 0
    @Namespace private var ns

    private var t: ThemedColors { ThemedColors(isDark: scheme == .dark) }
    private func s(_ key: LocalizedKey) -> String { localization.str(key) }

    var body: some View {
        ZStack {
            t.bg.ignoresSafeArea()

            // Background blobs
            Circle()
                .fill(AppTheme.accent.opacity(scheme == .dark ? 0.12 : 0.18))
                .frame(width: 340, height: 340).blur(radius: 80).offset(x: -80, y: -200)
            Circle()
                .fill(Color(hex: "#FF6B9D").opacity(scheme == .dark ? 0.08 : 0.12))
                .frame(width: 280, height: 280).blur(radius: 70).offset(x: 120, y: 300)

            VStack(spacing: 0) {
                // Logo
                VStack(spacing: 6) {
                    HStack(spacing: 0) {
                        Text("Fin")
                            .font(.system(size: 38, weight: .black, design: .rounded))
                            .foregroundColor(t.textPrimary)
                        Text("cora")
                            .font(.system(size: 38, weight: .black, design: .rounded))
                            .foregroundStyle(AppTheme.accentGradient)
                    }
                    Text("Your intelligent finance companion")
                        .font(.subheadline)
                        .foregroundColor(t.textMuted)
                }
                .padding(.top, 60)

                // Tab Selector
                HStack(spacing: 4) {
                    ForEach([s(.login), s(.register)].indices, id: \.self) { idx in
                        let title = [s(.login), s(.register)][idx]
                        Button(action: {
                            withAnimation(.spring(response: 0.4, dampingFraction: 0.75)) { selectedTab = idx }
                        }) {
                            Text(title)
                                .font(.system(size: 15, weight: .semibold))
                                .foregroundColor(selectedTab == idx ? .white : t.textMuted)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 11)
                                .background {
                                    if selectedTab == idx {
                                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                                            .fill(AppTheme.accentGradient)
                                            .matchedGeometryEffect(id: "AuthTab", in: ns)
                                            .shadow(color: AppTheme.accent.opacity(0.4), radius: 8, y: 4)
                                    }
                                }
                        }
                        .buttonStyle(PressEffect())
                    }
                }
                .padding(4)
                .background(t.surface2)
                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                .padding(.horizontal, 24)
                .padding(.top, 36)

                // Form Content
                ZStack {
                    if selectedTab == 0 {
                        LoginFormView()
                            .transition(.asymmetric(
                                insertion: .move(edge: .leading).combined(with: .opacity),
                                removal:   .move(edge: .trailing).combined(with: .opacity)
                            ))
                    } else {
                        RegisterFormView()
                            .transition(.asymmetric(
                                insertion: .move(edge: .trailing).combined(with: .opacity),
                                removal:   .move(edge: .leading).combined(with: .opacity)
                            ))
                    }
                }
                .animation(.spring(response: 0.4, dampingFraction: 0.85), value: selectedTab)

                Spacer()
            }
        }
    }
}

// MARK: - Login Form
struct LoginFormView: View {
    @EnvironmentObject var authManager: AuthManager
    @EnvironmentObject var localization: LocalizationManager
    @Environment(\.colorScheme) var scheme
    @State private var email       = ""
    @State private var password    = ""
    @State private var isLoading   = false
    @State private var errorMessage: String?
    @State private var showForgot  = false

    private var t: ThemedColors { ThemedColors(isDark: scheme == .dark) }
    private func s(_ key: LocalizedKey) -> String { localization.str(key) }

    var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 20) {
                VStack(spacing: 16) {
                    AuthField(icon: "envelope.fill", placeholder: s(.email), text: $email, isSecure: false)
                        .keyboardType(.emailAddress)
                        .textInputAutocapitalization(.never)
                    AuthField(icon: "lock.fill", placeholder: s(.password), text: $password, isSecure: true)
                }
                .padding(.top, 8)

                // Forgot Password
                HStack {
                    Spacer()
                    Button(s(.forgotPassword)) { showForgot = true }
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(AppTheme.accent)
                }

                // Error
                if let err = errorMessage {
                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.circle.fill")
                        Text(err)
                    }
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(AppTheme.red)
                    .padding(12)
                    .background(AppTheme.red.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .transition(.scale.combined(with: .opacity))
                }

                // Sign In Button
                Button(action: handleLogin) {
                    ZStack {
                        RoundedRectangle(cornerRadius: AppTheme.radiusMD, style: .continuous)
                            .fill(AppTheme.accentGradient)
                            .shadow(color: AppTheme.accent.opacity(0.35), radius: 12, y: 5)
                        if isLoading {
                            ProgressView().tint(.white)
                        } else {
                            Text(s(.login))
                                .font(.system(size: 16, weight: .bold))
                                .foregroundColor(.white)
                        }
                    }
                    .frame(height: 54)
                }
                .buttonStyle(PressEffect())
                .disabled(isLoading || email.isEmpty || password.isEmpty)
                .opacity(email.isEmpty || password.isEmpty ? 0.5 : 1)

                // Demo hint
                Text("Demo: demo@finapp.app / password123")
                    .font(.system(size: 11))
                    .foregroundColor(t.textMuted)
                    .padding(.top, 4)
            }
            .padding(.horizontal, 24)
            .padding(.top, 24)
        }
        .sheet(isPresented: $showForgot) {
            ForgotPasswordView()
                .environmentObject(localization)
        }
    }

    private func handleLogin() {
        withAnimation { errorMessage = nil; isLoading = true }
        try? authManager.register(fullName: "Demo User", email: "demo@finapp.app",
                                  password: "password123", confirm: "password123")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
            do {
                try authManager.login(email: email, password: password)
            } catch {
                withAnimation { errorMessage = error.localizedDescription }
            }
            withAnimation { isLoading = false }
        }
    }
}

// MARK: - Register Form
struct RegisterFormView: View {
    @EnvironmentObject var authManager: AuthManager
    @EnvironmentObject var localization: LocalizationManager
    @Environment(\.colorScheme) var scheme
    @State private var fullName    = ""
    @State private var email       = ""
    @State private var password    = ""
    @State private var confirm     = ""
    @State private var isLoading   = false
    @State private var errorMessage: String?

    private var t: ThemedColors { ThemedColors(isDark: scheme == .dark) }
    private func s(_ key: LocalizedKey) -> String { localization.str(key) }

    var passwordValid: Bool  { password.count >= 8 }
    var passwordsMatch: Bool { !confirm.isEmpty && password == confirm }
    var formValid: Bool      { !fullName.isEmpty && !email.isEmpty && passwordValid && passwordsMatch }

    var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 16) {
                AuthField(icon: "person.fill",    placeholder: s(.fullName),        text: $fullName,  isSecure: false)
                    .textInputAutocapitalization(.words)
                AuthField(icon: "envelope.fill",  placeholder: s(.email),           text: $email,     isSecure: false)
                    .keyboardType(.emailAddress).textInputAutocapitalization(.never)
                AuthField(icon: "lock.fill",      placeholder: s(.password),        text: $password,  isSecure: true)
                AuthField(icon: "lock.rotation",  placeholder: s(.confirmPassword), text: $confirm,   isSecure: true)

                // Requirements
                if !password.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        RequirementPill(text: "At least 8 characters", met: passwordValid)
                        RequirementPill(text: "Passwords match",       met: passwordsMatch)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }

                if let err = errorMessage {
                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.circle.fill")
                        Text(err)
                    }
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(AppTheme.red)
                    .padding(12).background(AppTheme.red.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .transition(.scale.combined(with: .opacity))
                }

                Button(action: handleRegister) {
                    ZStack {
                        RoundedRectangle(cornerRadius: AppTheme.radiusMD, style: .continuous)
                            .fill(formValid
                                  ? AppTheme.greenGradient
                                  : LinearGradient(colors: [t.surface2], startPoint: .leading, endPoint: .trailing))
                            .shadow(color: formValid ? AppTheme.green.opacity(0.3) : .clear, radius: 12, y: 5)
                        if isLoading {
                            ProgressView().tint(.white)
                        } else {
                            Text(s(.createAccount))
                                .font(.system(size: 16, weight: .bold))
                                .foregroundColor(.white)
                        }
                    }
                    .frame(height: 54)
                }
                .buttonStyle(PressEffect())
                .disabled(!formValid || isLoading)

                Text("By registering you agree to our Terms & Privacy Policy")
                    .font(.system(size: 11))
                    .foregroundColor(t.textMuted)
                    .multilineTextAlignment(.center)
            }
            .padding(.horizontal, 24)
            .padding(.top, 24)
        }
        .animation(.spring(response: 0.4, dampingFraction: 0.85), value: password.isEmpty)
    }

    private func handleRegister() {
        withAnimation { errorMessage = nil; isLoading = true }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
            do {
                try authManager.register(fullName: fullName, email: email,
                                         password: password, confirm: confirm)
            } catch {
                withAnimation { errorMessage = error.localizedDescription }
            }
            withAnimation { isLoading = false }
        }
    }
}

// MARK: - Forgot Password
struct ForgotPasswordView: View {
    @EnvironmentObject var localization: LocalizationManager
    @Environment(\.dismiss) private var dismiss
    @Environment(\.colorScheme) var scheme
    @State private var email     = ""
    @State private var isLoading = false
    @State private var sent      = false

    private var t: ThemedColors { ThemedColors(isDark: scheme == .dark) }
    private func s(_ key: LocalizedKey) -> String { localization.str(key) }

    var body: some View {
        ZStack {
            t.bg.ignoresSafeArea()
            VStack(spacing: 28) {
                ZStack {
                    Circle().fill(AppTheme.yellow.opacity(0.15)).frame(width: 80, height: 80)
                    Image(systemName: "key.fill").font(.system(size: 30)).foregroundStyle(AppTheme.yellow)
                }
                VStack(spacing: 8) {
                    Text("Reset Password")
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(t.textPrimary)
                    Text("Enter your email and we'll send reset instructions")
                        .font(.subheadline)
                        .foregroundColor(t.textMuted)
                        .multilineTextAlignment(.center)
                }

                AuthField(icon: "envelope.fill", placeholder: s(.email), text: $email, isSecure: false)
                    .keyboardType(.emailAddress).textInputAutocapitalization(.never)

                if sent {
                    HStack(spacing: 8) {
                        Image(systemName: "checkmark.circle.fill").foregroundColor(AppTheme.green)
                        Text("Instructions sent! Check your inbox.")
                    }
                    .font(.system(size: 13, weight: .medium)).foregroundColor(AppTheme.green)
                    .padding(12).background(AppTheme.green.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .transition(.scale.combined(with: .opacity))
                }

                Button(action: {
                    isLoading = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                        isLoading = false
                        withAnimation { sent = true }
                    }
                }) {
                    ZStack {
                        RoundedRectangle(cornerRadius: AppTheme.radiusMD)
                            .fill(email.isEmpty
                                  ? AnyShapeStyle(t.surface2)
                                  : AnyShapeStyle(LinearGradient(colors: [AppTheme.yellow, AppTheme.orange],
                                                                 startPoint: .leading, endPoint: .trailing)))
                        if isLoading {
                            ProgressView().tint(.white)
                        } else {
                            Text("Send Instructions")
                                .font(.system(size: 16, weight: .bold))
                                .foregroundColor(.white)
                        }
                    }.frame(height: 54)
                }
                .buttonStyle(PressEffect())
                .disabled(email.isEmpty || isLoading)

                Button("Back to Sign In") { dismiss() }
                    .font(.system(size: 14, weight: .medium)).foregroundColor(AppTheme.accent)
                Spacer()
            }
            .padding(.horizontal, 28)
            .padding(.top, 60)
        }
        .animation(.spring(response: 0.4), value: sent)
    }
}

// MARK: - AuthField (theme-aware)
struct AuthField: View {
    let icon: String
    let placeholder: String
    @Binding var text: String
    let isSecure: Bool
    @FocusState private var focused: Bool
    @State private var showPassword = false
    @Environment(\.colorScheme) var scheme

    private var t: ThemedColors { ThemedColors(isDark: scheme == .dark) }

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 15))
                .foregroundColor(focused ? AppTheme.accent : t.textMuted)
                .frame(width: 20)
                .animation(.easeInOut(duration: 0.2), value: focused)

            if isSecure && !showPassword {
                SecureField(placeholder, text: $text)
                    .focused($focused)
                    .foregroundColor(t.textPrimary)
                    .font(.system(size: 15))
            } else {
                TextField(placeholder, text: $text)
                    .focused($focused)
                    .foregroundColor(t.textPrimary)
                    .font(.system(size: 15))
                    .autocorrectionDisabled()
            }

            if isSecure {
                Button(action: { showPassword.toggle() }) {
                    Image(systemName: showPassword ? "eye.slash" : "eye")
                        .font(.system(size: 14))
                        .foregroundColor(t.textMuted)
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 16)
        .background(t.surface)
        .clipShape(RoundedRectangle(cornerRadius: AppTheme.radiusMD, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: AppTheme.radiusMD, style: .continuous)
                .stroke(focused ? AppTheme.accent : t.border, lineWidth: focused ? 1.5 : 1)
                .animation(.easeInOut(duration: 0.2), value: focused)
        )
    }
}

// MARK: - RequirementPill
struct RequirementPill: View {
    let text: String
    let met: Bool
    @Environment(\.colorScheme) var scheme
    private var t: ThemedColors { ThemedColors(isDark: scheme == .dark) }

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: met ? "checkmark.circle.fill" : "circle")
                .font(.system(size: 13))
                .foregroundColor(met ? AppTheme.green : t.textMuted)
            Text(text)
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(met ? AppTheme.green : t.textMuted)
        }
        .animation(.spring(response: 0.3), value: met)
    }
}
