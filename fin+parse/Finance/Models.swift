// Models.swift — Data models, Auth, SharedDataManager

import SwiftUI
import Combine

// MARK: - User Model
struct User: Codable, Equatable {
    var id: String
    var fullName: String
    var email: String
    var passwordHash: String   // In production: bcrypt. Here: simple hash simulation.
    var createdAt: Date

    var initials: String {
        fullName.split(separator: " ")
            .prefix(2)
            .compactMap { $0.first.map(String.init) }
            .joined()
            .uppercased()
    }
    var firstName: String { fullName.split(separator: " ").first.map(String.init) ?? fullName }
}

// MARK: - Auth Manager (Backend simulation with UserDefaults persistence)
class AuthManager: ObservableObject {
    @Published var currentUser: User?
    @Published var isAuthenticated = false

    private let usersKey = "fincora_users"
    private let sessionKey = "fincora_session"

    init() { restoreSession() }

    // --- Persistence helpers ---
    private func loadUsers() -> [User] {
        guard let data = UserDefaults.standard.data(forKey: usersKey),
              let users = try? JSONDecoder().decode([User].self, from: data) else { return [] }
        return users
    }
    private func saveUsers(_ users: [User]) {
        if let data = try? JSONEncoder().encode(users) {
            UserDefaults.standard.set(data, forKey: usersKey)
        }
    }
    private func restoreSession() {
        guard let data = UserDefaults.standard.data(forKey: sessionKey),
              let user = try? JSONDecoder().decode(User.self, from: data) else { return }
        currentUser = user
        isAuthenticated = true
    }
    private func saveSession(_ user: User) {
        if let data = try? JSONEncoder().encode(user) {
            UserDefaults.standard.set(data, forKey: sessionKey)
        }
    }
    private func clearSession() {
        UserDefaults.standard.removeObject(forKey: sessionKey)
    }

    // Naive hash (replace with CryptoKit SHA256 in production)
    private func hashPassword(_ pw: String) -> String { String(pw.hashValue) }

    // --- Auth Operations ---
    enum AuthError: LocalizedError {
        case emailTaken, invalidCredentials, weakPassword, emptyFields, passwordMismatch
        var errorDescription: String? {
            switch self {
            case .emailTaken:          return "An account with this email already exists."
            case .invalidCredentials:  return "Incorrect email or password."
            case .weakPassword:        return "Password must be at least 8 characters."
            case .emptyFields:         return "Please fill in all fields."
            case .passwordMismatch:    return "Passwords do not match."
            }
        }
    }

    func register(fullName: String, email: String, password: String, confirm: String) throws {
        guard !fullName.isEmpty, !email.isEmpty, !password.isEmpty else { throw AuthError.emptyFields }
        guard password == confirm else { throw AuthError.passwordMismatch }
        guard password.count >= 8 else { throw AuthError.weakPassword }
        var users = loadUsers()
        guard !users.contains(where: { $0.email.lowercased() == email.lowercased() }) else { throw AuthError.emailTaken }

        let newUser = User(id: UUID().uuidString, fullName: fullName, email: email,
                          passwordHash: hashPassword(password), createdAt: Date())
        users.append(newUser)
        saveUsers(users)
        // Auto-login after register
        currentUser = newUser
        isAuthenticated = true
        saveSession(newUser)
    }

    func login(email: String, password: String) throws {
        guard !email.isEmpty, !password.isEmpty else { throw AuthError.emptyFields }
        let users = loadUsers()
        guard let user = users.first(where: {
            $0.email.lowercased() == email.lowercased() && $0.passwordHash == hashPassword(password)
        }) else { throw AuthError.invalidCredentials }
        currentUser = user
        isAuthenticated = true
        saveSession(user)
    }

    func logout() {
        currentUser = nil
        isAuthenticated = false
        clearSession()
    }
}

// MARK: - Transaction
struct Transaction: Identifiable, Hashable, Codable {
    var id = UUID()
    var title: String
    var amount: Double
    var category: TransactionCategory
    var date: Date
    var isExpense: Bool
    var note: String = ""

    var formattedAmount: String {
        let sign = isExpense ? "-" : "+"
        return "\(sign)₸\(String(format: "%.2f", abs(amount)))"
    }
    var formattedDate: String {
        let f = DateFormatter(); f.dateFormat = "MMM d, yyyy"
        return f.string(from: date)
    }
    var shortDate: String {
        let f = DateFormatter(); f.dateFormat = "MMM d"
        return f.string(from: date)
    }
}

// MARK: - Transaction Category
enum TransactionCategory: String, CaseIterable, Codable {
    case salary = "Salary"
    case freelance = "Freelance"
    case investment = "Investment"
    case transfer = "Transfer"
    case shopping = "Shopping"
    case food = "Food"
    case housing = "Housing"
    case transport = "Transport"
    case health = "Health"
    case entertainment = "Entertainment"
    case utilities = "Utilities"
    case subscriptions = "Subscriptions"
    case other = "Other"

    var isIncomeCategory: Bool {
        [.salary, .freelance, .investment].contains(self)
    }
    var icon: String {
            switch self {
            case .salary:        return "briefcase.fill"
            case .freelance:     return "laptopcomputer"
            case .investment:    return "chart.line.uptrend.xyaxis"
            case .transfer:      return "arrow.left.arrow.right" // Added
            case .shopping:      return "bag.fill"
            case .food:          return "fork.knife"
            case .housing:       return "house.fill"
            case .transport:     return "car.fill"
            case .health:        return "heart.fill"
            case .entertainment: return "play.fill"
            case .utilities:     return "bolt.fill"
            case .subscriptions: return "tv.fill"
            case .other:         return "ellipsis.circle.fill"
            }
        }
    var color: Color {
            switch self {
            case .salary:        return AppTheme.green
            case .freelance:     return AppTheme.accent
            case .investment:    return AppTheme.yellow
            case .transfer:      return AppTheme.textMuted // Added (grayish for neutrality)
            case .shopping:      return AppTheme.orange
            case .food:          return Color(hex: "#FF6B9D")
            case .housing:       return Color(hex: "#F7B731")
            case .transport:     return AppTheme.teal
            case .health:        return Color(hex: "#FF5F6D")
            case .entertainment: return AppTheme.accent
            case .utilities:     return AppTheme.yellow
            case .subscriptions: return Color(hex: "#A78BFA")
            case .other:         return AppTheme.textMuted
            }
        }
    var softColor: Color { color.opacity(0.18) }
}

// MARK: - Budget
struct Budget: Identifiable, Codable {
    var id = UUID()
    var category: TransactionCategory
    var limit: Double
    var period: String = "Monthly"

    var percentUsed: Double { 0 } // Computed in SharedDataManager
}

// MARK: - Savings Goal
struct SavingsGoal: Identifiable, Codable {
    var id = UUID()
    var name: String
    var emoji: String
    var targetAmount: Double
    var savedAmount: Double
    var deadline: Date
    var colorHex: String

    var progress: Double { min(savedAmount / targetAmount, 1.0) }
    var progressPercent: Int { Int(progress * 100) }
    var color: Color { Color(hex: colorHex) }
    var formattedDeadline: String {
        let f = DateFormatter(); f.dateFormat = "MMM yyyy"
        return f.string(from: deadline)
    }
}

// MARK: - Trend
enum TrendDirection { case up, down }

// MARK: - Category Spending
struct CategorySpending: Identifiable {
    var id = UUID()
    let category: TransactionCategory
    let amount: Double
    let percentage: Double
}

// MARK: - Monthly Data
struct MonthlyData: Identifiable {
    var id = UUID()
    let month: String
    let income: Double
    let expenses: Double
    var net: Double { income - expenses }
}

// MARK: - Message
struct Message: Identifiable, Hashable {
    let id = UUID()
    let text: String
    let isFromUser: Bool
    let timestamp: Date

    init(_ text: String, isFromUser: Bool = false, timestamp: Date = Date()) {
        self.text = text; self.isFromUser = isFromUser; self.timestamp = timestamp
    }
}

// MARK: - Shared Data Manager
class SharedDataManager: ObservableObject {
    @Published var transactions: [Transaction] = []
    @Published var budgets: [Budget] = []
    @Published var goals: [SavingsGoal] = []

    private let txnKey = "fincora_transactions"
    private let budgetKey = "fincora_budgets"
    private let goalsKey = "fincora_goals"

    init() { loadAll() }

    // --- Persistence ---
    private func loadAll() {
        transactions = decode([Transaction].self, key: txnKey) ?? Self.sampleTransactions
        budgets      = decode([Budget].self,      key: budgetKey) ?? Self.sampleBudgets
        goals        = decode([SavingsGoal].self, key: goalsKey)  ?? Self.sampleGoals
    }
    func saveAll() {
        encode(transactions, key: txnKey)
        encode(budgets, key: budgetKey)
        encode(goals, key: goalsKey)
    }
    private func decode<T: Decodable>(_ type: T.Type, key: String) -> T? {
        guard let data = UserDefaults.standard.data(forKey: key) else { return nil }
        return try? JSONDecoder().decode(type, from: data)
    }
    private func encode<T: Encodable>(_ value: T, key: String) {
        if let data = try? JSONEncoder().encode(value) { UserDefaults.standard.set(data, forKey: key) }
    }

    // --- CRUD ---
    func addTransaction(_ t: Transaction) { transactions.insert(t, at: 0); saveAll() }
    func deleteTransaction(_ t: Transaction) { transactions.removeAll { $0.id == t.id }; saveAll() }
    func addGoal(_ g: SavingsGoal) { goals.append(g); saveAll() }
    func addBudget(_ b: Budget) {
        if let i = budgets.firstIndex(where: { $0.category == b.category }) {
            budgets[i] = b
        } else { budgets.append(b) }
        saveAll()
    }

    // --- Computed ---
    var totalIncome: Double   { transactions.filter { !$0.isExpense }.reduce(0) { $0 + $1.amount } }
    var totalExpenses: Double { transactions.filter { $0.isExpense }.reduce(0) { $0 + $1.amount } }
    var balance: Double       { totalIncome - totalExpenses }
    var savingsRate: Double   { totalIncome > 0 ? (totalIncome - totalExpenses) / totalIncome * 100 : 0 }

    var thisMonthExpenses: Double {
        let cal = Calendar.current; let now = Date()
        return transactions.filter { $0.isExpense && cal.isDate($0.date, equalTo: now, toGranularity: .month) }
            .reduce(0) { $0 + $1.amount }
    }
    var lastMonthExpenses: Double { thisMonthExpenses * 0.9 }
    var monthlyTrend: TrendDirection { thisMonthExpenses > lastMonthExpenses ? .up : .down }

    var categorySpending: [CategorySpending] {
        let expenses = transactions.filter { $0.isExpense }
        let total = totalExpenses
        var map: [TransactionCategory: Double] = [:]
        for t in expenses { map[t.category, default: 0] += t.amount }
        return map.map { CategorySpending(category: $0.key, amount: $0.value, percentage: total > 0 ? $0.value / total : 0) }
            .sorted { $0.amount > $1.amount }
    }

    func spent(for category: TransactionCategory) -> Double {
        transactions.filter { $0.isExpense && $0.category == category }.reduce(0) { $0 + $1.amount }
    }
    var totalBudgetLimit: Double {
        budgets.reduce(0.0) { $0 + $1.limit }
    }

    var totalBudgetSpent: Double {
        budgets.reduce(0.0) { $0 + spent(for: $1.category) }
    }
    var monthlyData: [MonthlyData] {
        let labels = ["Jul","Aug","Sep","Oct","Nov","Dec","Jan"]
        let incomes  = [4800.0, 5200, 5100, 5900, 5600, 6200, totalIncome > 0 ? totalIncome : 6200]
        let expenses = [3600.0, 3900, 3500, 4200, 3800, 3840, totalExpenses > 0 ? totalExpenses : 3840]
        return zip(labels, zip(incomes, expenses)).map { MonthlyData(month: $0.0, income: $0.1.0, expenses: $0.1.1) }
    }

    var weeklySpend: [(String, Double)] {
        [("Mon",42),("Tue",38),("Wed",65),("Thu",55),("Fri",120),("Sat",180),("Sun",95)]
    }

    // --- Sample Data ---
    static let sampleTransactions: [Transaction] = [
        Transaction(title: "Monthly Salary",    amount: 4500,   category: .salary,        date: daysAgo(2),  isExpense: false),
        Transaction(title: "Freelance Project", amount: 1700,   category: .freelance,     date: daysAgo(5),  isExpense: false),
        Transaction(title: "Investment Return", amount: 340,    category: .investment,    date: daysAgo(10), isExpense: false),
        Transaction(title: "Rent",              amount: 1800,   category: .housing,       date: daysAgo(3),  isExpense: true),
        Transaction(title: "Whole Foods",       amount: 124.50, category: .food,          date: daysAgo(4),  isExpense: true),
        Transaction(title: "Netflix",           amount: 15.99,  category: .subscriptions, date: daysAgo(6),  isExpense: true),
        Transaction(title: "Electricity Bill",  amount: 89,     category: .utilities,     date: daysAgo(7),  isExpense: true),
        Transaction(title: "Gym Membership",    amount: 49,     category: .health,        date: daysAgo(8),  isExpense: true),
        Transaction(title: "Amazon",            amount: 54.30,  category: .shopping,      date: daysAgo(9),  isExpense: true),
        Transaction(title: "Uber",              amount: 22.40,  category: .transport,     date: daysAgo(11), isExpense: true),
        Transaction(title: "Spotify",           amount: 9.99,   category: .subscriptions, date: daysAgo(12), isExpense: true),
        Transaction(title: "Restaurant",        amount: 67.80,  category: .food,          date: daysAgo(14), isExpense: true),
    ]
    static let sampleBudgets: [Budget] = [
        Budget(category: .food,          limit: 600),
        Budget(category: .entertainment, limit: 200),
        Budget(category: .transport,     limit: 300),
        Budget(category: .shopping,      limit: 400),
        Budget(category: .health,        limit: 150),
        Budget(category: .utilities,     limit: 250),
    ]
    static let sampleGoals: [SavingsGoal] = [
        SavingsGoal(name: "Emergency Fund",   emoji: "🛡️", targetAmount: 10000, savedAmount: 6800,  deadline: daysFromNow(180), colorHex: "#7C6FCD"),
        SavingsGoal(name: "Japan Vacation",   emoji: "✈️", targetAmount: 4500,  savedAmount: 1920,  deadline: daysFromNow(120), colorHex: "#FF6B9D"),
        SavingsGoal(name: "New MacBook",      emoji: "💻", targetAmount: 2500,  savedAmount: 2100,  deadline: daysFromNow(30),  colorHex: "#34C97B"),
        SavingsGoal(name: "Investment Fund",  emoji: "📈", targetAmount: 50000, savedAmount: 18400, deadline: daysFromNow(365), colorHex: "#FFB730"),
    ]
    private static func daysAgo(_ n: Int) -> Date { Calendar.current.date(byAdding: .day, value: -n, to: Date())! }
    private static func daysFromNow(_ n: Int) -> Date { Calendar.current.date(byAdding: .day, value: n, to: Date())! }
}

// MARK: - Dashboard API Response Models

/// GET /api/dashboard/overview
struct DashboardOverview: Codable {
    let totalSpent: Double
    let totalIncome: Double
    let balance: Double
    let transactionCount: Int
    let topCategory: String?
    let thisMonthSpent: Double
    let lastMonthSpent: Double
    let trend: String                // "up" | "down"

    enum CodingKeys: String, CodingKey {
        case totalSpent = "total_spent"
        case totalIncome = "total_income"
        case balance
        case transactionCount = "transaction_count"
        case topCategory = "top_category"
        case thisMonthSpent = "this_month_spent"
        case lastMonthSpent = "last_month_spent"
        case trend
    }

    var trendDirection: TrendDirection { trend == "up" ? .up : .down }
}

/// One row inside GET /api/dashboard/categories
struct APICategoryItem: Codable, Identifiable {
    var id: String { category }
    let category: String
    let amount: Double
    let count: Int
    let percentage: Double

    /// Map API category string to the app's TransactionCategory enum.
    var appCategory: TransactionCategory {
        TransactionCategory(rawValue: category) ?? .other
    }
}

/// GET /api/dashboard/categories
struct APICategoryResponse: Codable {
    let categories: [APICategoryItem]
    let total: Double
}

/// One row inside GET /api/dashboard/banks
struct APIBankItem: Codable, Identifiable {
    var id: String { bank }
    let bank: String
    let amount: Double
    let count: Int
    let average: Double
    let percentage: Double

    /// Short display name.
    var shortName: String {
        if bank.lowercased().contains("kaspi") { return "Kaspi" }
        if bank.lowercased().contains("freedom") { return "Freedom" }
        return bank
    }
}

/// GET /api/dashboard/banks
struct APIBankResponse: Codable {
    let banks: [APIBankItem]
    let total: Double
}

/// One row inside GET /api/dashboard/operations
struct APIOperationItem: Codable, Identifiable {
    var id: String { operation }
    let operation: String
    let amount: Double
    let count: Int
    let percentage: Double
}

/// GET /api/dashboard/operations
struct APIOperationResponse: Codable {
    let operations: [APIOperationItem]
    let total: Double
}

/// One month in GET /api/dashboard/trend
struct APITrendMonth: Codable, Identifiable {
    var id: String { month }
    let month: String
    let expenses: Double
    let income: Double

    /// Human-readable short label, e.g. "Nov 24"
    var label: String {
        let parts = month.split(separator: "-")
        guard parts.count == 2,
              let m = Int(parts[1]) else { return month }
        let names = ["","Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]
        let yr = String(parts[0].suffix(2))
        return m >= 1 && m <= 12 ? "\(names[m]) \(yr)" : month
    }
}

/// GET /api/dashboard/trend
struct APITrendResponse: Codable {
    let trend: [APITrendMonth]
}

// MARK: - Dashboard Filter State

/// Which time window the user picked.
enum DashboardPeriod: String, CaseIterable, Identifiable {
    case oneMonth  = "1m"
    case threeMonths = "3m"
    case sixMonths = "6m"
    case oneYear   = "1y"
    case all       = "all"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .oneMonth:    return "1M"
        case .threeMonths: return "3M"
        case .sixMonths:   return "6M"
        case .oneYear:     return "1Y"
        case .all:         return "All"
        }
    }
}

/// Bank filter choices.
enum DashboardBankFilter: String, CaseIterable, Identifiable {
    case all     = ""
    case kaspi   = "kaspi"
    case freedom = "freedom"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .all:     return "All"
        case .kaspi:   return "Kaspi"
        case .freedom: return "Freedom"
        }
    }
}

/// Transaction type filter choices.
enum DashboardTypeFilter: String, CaseIterable, Identifiable {
    case all       = ""
    case transfer  = "transfer"
    case expenses  = "expenses"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .all:      return "All"
        case .transfer: return "Transfers"
        case .expenses: return "Expenses"
        }
    }
}
