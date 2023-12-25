
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_5_13_24_16>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001700: F4080100 F8000000
	s_ashr_i32 s2, s13, 31                                     // 000000001708: 86029F0D
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_lshr_b32 s2, s2, 28                                      // 000000001714: 85029C02
	s_mulk_i32 s14, 0x180                                      // 000000001718: B80E0180
	s_add_i32 s9, s13, s2                                      // 00000000171C: 8109020D
	s_mul_i32 s2, s15, 0x1380                                  // 000000001720: 9602FF0F 00001380
	s_and_b32 s3, s9, -16                                      // 000000001728: 8B03D009
	s_add_i32 s2, s2, s14                                      // 00000000172C: 81020E02
	s_sub_i32 s8, s13, s3                                      // 000000001730: 8188030D
	s_ashr_i32 s10, s9, 4                                      // 000000001734: 860A8409
	s_add_i32 s2, s2, s8                                       // 000000001738: 81020802
	s_mul_i32 s8, s15, 24                                      // 00000000173C: 9608980F
	s_add_i32 s2, s2, s3                                       // 000000001740: 81020302
	v_mov_b32_e32 v1, 0                                        // 000000001744: 7E020280
	s_ashr_i32 s3, s2, 31                                      // 000000001748: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 00000000174C: BF8704D9
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001750: 84828202
	s_waitcnt lgkmcnt(0)                                       // 000000001754: BF89FC07
	s_add_u32 s6, s6, s2                                       // 000000001758: 80060206
	s_addc_u32 s7, s7, s3                                      // 00000000175C: 82070307
	s_ashr_i32 s9, s8, 31                                      // 000000001760: 86099F08
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001764: 84888208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001768: BF8704B9
	s_add_u32 s8, s0, s8                                       // 00000000176C: 80080800
	s_addc_u32 s9, s1, s9                                      // 000000001770: 82090901
	s_ashr_i32 s11, s10, 31                                    // 000000001774: 860B9F0A
	s_lshl_b64 s[0:1], s[10:11], 2                             // 000000001778: 8480820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000177C: BF870009
	s_add_u32 s0, s8, s0                                       // 000000001780: 80000008
	s_addc_u32 s1, s9, s1                                      // 000000001784: 82010109
	s_load_b32 s6, s[6:7], null                                // 000000001788: F4000183 F8000000
	s_load_b32 s0, s[0:1], null                                // 000000001790: F4000000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001798: BF89FC07
	v_add_f32_e64 v0, s6, s0                                   // 00000000179C: D5030000 00000006
	s_add_u32 s0, s4, s2                                       // 0000000017A4: 80000204
	s_addc_u32 s1, s5, s3                                      // 0000000017A8: 82010305
	global_store_b32 v1, v0, s[0:1]                            // 0000000017AC: DC6A0000 00000001
	s_nop 0                                                    // 0000000017B4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017B8: BFB60003
	s_endpgm                                                   // 0000000017BC: BFB00000
