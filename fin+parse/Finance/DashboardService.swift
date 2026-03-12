// DashboardService.swift
// Calls the Flask /api/dashboard/* endpoints and publishes results for SwiftUI views.

import Foundation
import Combine

@MainActor
final class DashboardService: ObservableObject {

    // MARK: - Filter state (drives all fetches)
    @Published var period: DashboardPeriod       = .all
    @Published var bankFilter: DashboardBankFilter = .all
    @Published var typeFilter: DashboardTypeFilter = .all

    // MARK: - Response data
    @Published var overview: DashboardOverview?
    @Published var categories: APICategoryResponse?
    @Published var banks: APIBankResponse?
    @Published var operations: APIOperationResponse?
    @Published var trend: APITrendResponse?
    @Published var isLoading = false
    @Published var errorMessage: String?

    private var cancellables = Set<AnyCancellable>()

    init() {
        // Re-fetch whenever any filter changes.
        Publishers.CombineLatest3($period, $bankFilter, $typeFilter)
            .debounce(for: .milliseconds(200), scheduler: RunLoop.main)
            .sink { [weak self] _ in
                Task { await self?.fetchAll() }
            }
            .store(in: &cancellables)
    }

    // MARK: - Public

    func fetchAll() async {
        isLoading = true
        errorMessage = nil

        async let o = fetch(DashboardOverview.self, path: "overview")
        async let c = fetch(APICategoryResponse.self, path: "categories")
        async let b = fetch(APIBankResponse.self, path: "banks")
        async let ops = fetch(APIOperationResponse.self, path: "operations")
        async let t = fetch(APITrendResponse.self, path: "trend")

        overview   = await o
        categories = await c
        banks      = await b
        operations = await ops
        trend      = await t

        isLoading = false
    }

    // MARK: - Private

    private func fetch<T: Decodable>(_ type: T.Type, path: String) async -> T? {
        guard let url = buildURL(path: path) else { return nil }
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                let code = (response as? HTTPURLResponse)?.statusCode ?? -1
                errorMessage = "Server returned \(code) for /\(path)"
                return nil
            }
            return try JSONDecoder().decode(type, from: data)
        } catch {
            errorMessage = error.localizedDescription
            return nil
        }
    }

    private func buildURL(path: String) -> URL? {
        var components = URLComponents(string: "\(APIConfig.baseURL)/api/dashboard/\(path)")
        var items: [URLQueryItem] = []

        if period != .all {
            items.append(URLQueryItem(name: "period", value: period.rawValue))
        }
        if bankFilter != .all {
            items.append(URLQueryItem(name: "bank", value: bankFilter.rawValue))
        }
        if typeFilter != .all {
            items.append(URLQueryItem(name: "type", value: typeFilter.rawValue))
        }
        if !items.isEmpty { components?.queryItems = items }
        return components?.url
    }
}
