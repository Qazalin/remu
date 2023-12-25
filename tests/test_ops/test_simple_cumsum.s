
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_256_256>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	v_mov_b32_e32 v0, 0                                        // 000000001708: 7E000280
	s_lshl_b32 s4, s15, 8                                      // 00000000170C: 8404880F
	s_mov_b32 s7, 0                                            // 000000001710: BE870080
	s_mov_b32 s8, 0                                            // 000000001714: BE880080
	s_waitcnt lgkmcnt(0)                                       // 000000001718: BF89FC07
	s_add_u32 s2, s2, -8                                       // 00000000171C: 8002C802
	s_addc_u32 s3, s3, -1                                      // 000000001720: 8203C103
	s_add_i32 s5, s14, s4                                      // 000000001724: 8105040E
	s_branch 10                                                // 000000001728: BFA0000A <r_4_256_256+0x54>
	s_waitcnt lgkmcnt(0)                                       // 00000000172C: BF89FC07
	v_add_f32_e32 v0, s9, v0                                   // 000000001730: 06000009
	s_add_i32 s8, s8, 4                                        // 000000001734: 81088408
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001738: BF870099
	s_cmpk_eq_i32 s8, 0x100                                    // 00000000173C: B1880100
	v_add_f32_e32 v0, s13, v0                                  // 000000001740: 0600000D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001744: BF870091
	v_add_f32_e32 v0, s12, v0                                  // 000000001748: 0600000C
	v_add_f32_e32 v0, s10, v0                                  // 00000000174C: 0600000A
	s_cbranch_scc1 106                                         // 000000001750: BFA2006A <r_4_256_256+0x1fc>
	s_add_i32 s11, s5, s8                                      // 000000001754: 810B0805
	s_add_i32 s10, s14, s8                                     // 000000001758: 810A080E
	s_add_i32 s6, s11, 0x301                                   // 00000000175C: 8106FF0B 00000301
	s_add_i32 s12, s10, -1                                     // 000000001764: 810CC10A
	s_ashr_i32 s9, s6, 31                                      // 000000001768: 86099F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000176C: BF870499
	s_lshr_b32 s9, s9, 22                                      // 000000001770: 85099609
	s_add_i32 s9, s6, s9                                       // 000000001774: 81090906
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001778: BF870499
	s_and_b32 s9, s9, 0xfffffc00                               // 00000000177C: 8B09FF09 FFFFFC00
	s_sub_i32 s6, s6, s9                                       // 000000001784: 81860906
	s_cmpk_lt_i32 s12, 0xfe                                    // 000000001788: B38C00FE
	s_cselect_b32 s9, -1, 0                                    // 00000000178C: 980980C1
	s_cmp_lt_i32 s6, 2                                         // 000000001790: BF048206
	s_cselect_b32 s12, -1, 0                                   // 000000001794: 980C80C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001798: BF870499
	s_or_b32 s9, s9, s12                                       // 00000000179C: 8C090C09
	s_and_b32 vcc_lo, exec_lo, s9                              // 0000000017A0: 8B6A097E
	s_mov_b32 s9, 0                                            // 0000000017A4: BE890080
	s_cbranch_vccnz 6                                          // 0000000017A8: BFA40006 <r_4_256_256+0xc4>
	s_lshl_b64 s[12:13], s[6:7], 2                             // 0000000017AC: 848C8206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017B0: BF870009
	s_add_u32 s12, s2, s12                                     // 0000000017B4: 800C0C02
	s_addc_u32 s13, s3, s13                                    // 0000000017B8: 820D0D03
	s_load_b32 s9, s[12:13], null                              // 0000000017BC: F4000246 F8000000
	s_add_i32 s6, s11, 0x302                                   // 0000000017C4: 8106FF0B 00000302
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017CC: BF870499
	s_ashr_i32 s12, s6, 31                                     // 0000000017D0: 860C9F06
	s_lshr_b32 s12, s12, 22                                    // 0000000017D4: 850C960C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017D8: BF870499
	s_add_i32 s12, s6, s12                                     // 0000000017DC: 810C0C06
	s_and_b32 s12, s12, 0xfffffc00                             // 0000000017E0: 8B0CFF0C FFFFFC00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 0000000017E8: BF8704D9
	s_sub_i32 s6, s6, s12                                      // 0000000017EC: 81860C06
	s_cmpk_lt_i32 s10, 0xfe                                    // 0000000017F0: B38A00FE
	s_cselect_b32 s12, -1, 0                                   // 0000000017F4: 980C80C1
	s_cmp_lt_i32 s6, 2                                         // 0000000017F8: BF048206
	s_cselect_b32 s13, -1, 0                                   // 0000000017FC: 980D80C1
	s_or_b32 s13, s12, s13                                     // 000000001800: 8C0D0D0C
	s_mov_b32 s12, 0                                           // 000000001804: BE8C0080
	s_and_b32 vcc_lo, exec_lo, s13                             // 000000001808: 8B6A0D7E
	s_mov_b32 s13, 0                                           // 00000000180C: BE8D0080
	s_cbranch_vccnz 6                                          // 000000001810: BFA40006 <r_4_256_256+0x12c>
	s_lshl_b64 s[16:17], s[6:7], 2                             // 000000001814: 84908206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001818: BF870009
	s_add_u32 s16, s2, s16                                     // 00000000181C: 80101002
	s_addc_u32 s17, s3, s17                                    // 000000001820: 82111103
	s_load_b32 s13, s[16:17], null                             // 000000001824: F4000348 F8000000
	s_add_i32 s6, s11, 0x303                                   // 00000000182C: 8106FF0B 00000303
	s_add_i32 s16, s10, 1                                      // 000000001834: 8110810A
	s_ashr_i32 s15, s6, 31                                     // 000000001838: 860F9F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000183C: BF870499
	s_lshr_b32 s15, s15, 22                                    // 000000001840: 850F960F
	s_add_i32 s15, s6, s15                                     // 000000001844: 810F0F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001848: BF870499
	s_and_b32 s15, s15, 0xfffffc00                             // 00000000184C: 8B0FFF0F FFFFFC00
	s_sub_i32 s6, s6, s15                                      // 000000001854: 81860F06
	s_cmpk_lt_i32 s16, 0xfe                                    // 000000001858: B39000FE
	s_cselect_b32 s15, -1, 0                                   // 00000000185C: 980F80C1
	s_cmp_lt_i32 s6, 2                                         // 000000001860: BF048206
	s_cselect_b32 s16, -1, 0                                   // 000000001864: 981080C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001868: BF870499
	s_or_b32 s15, s15, s16                                     // 00000000186C: 8C0F100F
	s_and_b32 vcc_lo, exec_lo, s15                             // 000000001870: 8B6A0F7E
	s_cbranch_vccnz 6                                          // 000000001874: BFA40006 <r_4_256_256+0x190>
	s_lshl_b64 s[16:17], s[6:7], 2                             // 000000001878: 84908206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000187C: BF870009
	s_add_u32 s16, s2, s16                                     // 000000001880: 80101002
	s_addc_u32 s17, s3, s17                                    // 000000001884: 82111103
	s_load_b32 s12, s[16:17], null                             // 000000001888: F4000308 F8000000
	s_add_i32 s6, s11, 0x304                                   // 000000001890: 8106FF0B 00000304
	s_add_i32 s10, s10, 2                                      // 000000001898: 810A820A
	s_ashr_i32 s11, s6, 31                                     // 00000000189C: 860B9F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000018A0: BF870499
	s_lshr_b32 s11, s11, 22                                    // 0000000018A4: 850B960B
	s_add_i32 s11, s6, s11                                     // 0000000018A8: 810B0B06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000018AC: BF870499
	s_and_b32 s11, s11, 0xfffffc00                             // 0000000018B0: 8B0BFF0B FFFFFC00
	s_sub_i32 s6, s6, s11                                      // 0000000018B8: 81860B06
	s_cmpk_lt_i32 s10, 0xfe                                    // 0000000018BC: B38A00FE
	s_cselect_b32 s10, -1, 0                                   // 0000000018C0: 980A80C1
	s_cmp_lt_i32 s6, 2                                         // 0000000018C4: BF048206
	s_cselect_b32 s11, -1, 0                                   // 0000000018C8: 980B80C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000018CC: BF870499
	s_or_b32 s10, s10, s11                                     // 0000000018D0: 8C0A0B0A
	s_and_b32 vcc_lo, exec_lo, s10                             // 0000000018D4: 8B6A0A7E
	s_mov_b32 s10, 0                                           // 0000000018D8: BE8A0080
	s_cbranch_vccnz 65427                                      // 0000000018DC: BFA4FF93 <r_4_256_256+0x2c>
	s_lshl_b64 s[10:11], s[6:7], 2                             // 0000000018E0: 848A8206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000018E4: BF870009
	s_add_u32 s10, s2, s10                                     // 0000000018E8: 800A0A02
	s_addc_u32 s11, s3, s11                                    // 0000000018EC: 820B0B03
	s_load_b32 s10, s[10:11], null                             // 0000000018F0: F4000285 F8000000
	s_branch 65420                                             // 0000000018F8: BFA0FF8C <r_4_256_256+0x2c>
	s_ashr_i32 s5, s4, 31                                      // 0000000018FC: 86059F04
	v_mov_b32_e32 v1, 0                                        // 000000001900: 7E020280
	s_lshl_b64 s[2:3], s[4:5], 2                               // 000000001904: 84828204
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001908: BF8704B9
	s_add_u32 s2, s0, s2                                       // 00000000190C: 80020200
	s_addc_u32 s3, s1, s3                                      // 000000001910: 82030301
	s_ashr_i32 s15, s14, 31                                    // 000000001914: 860F9F0E
	s_lshl_b64 s[0:1], s[14:15], 2                             // 000000001918: 8480820E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000191C: BF870009
	s_add_u32 s0, s2, s0                                       // 000000001920: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001924: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 000000001928: DC6A0000 00000001
	s_nop 0                                                    // 000000001930: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001934: BFB60003
	s_endpgm                                                   // 000000001938: BFB00000
