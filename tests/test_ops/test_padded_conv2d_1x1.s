
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_4_15_32_3>:
	s_ashr_i32 s2, s13, 31                                     // 000000001700: 86029F0D
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_lshr_b32 s2, s2, 27                                      // 00000000170C: 85029B02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001710: BF870499
	s_add_i32 s3, s13, s2                                      // 000000001714: 8103020D
	s_and_b32 s2, s3, 0xffffffe0                               // 000000001718: 8B02FF03 FFFFFFE0
	s_ashr_i32 s12, s3, 5                                      // 000000001720: 860C8503
	s_sub_i32 s2, s13, s2                                      // 000000001724: 8182020D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001728: BF870499
	s_add_i32 s8, s2, -2                                       // 00000000172C: 8108C202
	s_cmp_lt_u32 s8, 28                                        // 000000001730: BF0A9C08
	s_load_b64 s[8:9], s[0:1], 0x10                            // 000000001734: F4040200 F8000010
	s_cselect_b32 s3, -1, 0                                    // 00000000173C: 980380C1
	s_sub_i32 s10, s13, 64                                     // 000000001740: 818AC00D
	s_mul_i32 s0, s15, 0x39c                                   // 000000001744: 9600FF0F 0000039C
	s_cmpk_lt_u32 s10, 0x160                                   // 00000000174C: B68A0160
	s_mul_i32 s10, s12, 28                                     // 000000001750: 960A9C0C
	s_cselect_b32 s11, -1, 0                                   // 000000001754: 980B80C1
	s_ashr_i32 s1, s0, 31                                      // 000000001758: 86019F00
	s_and_b32 s13, s11, s3                                     // 00000000175C: 8B0D030B
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001760: 84808200
	v_cndmask_b32_e64 v0, 0, 1, s13                            // 000000001764: D5010000 00350280
	s_waitcnt lgkmcnt(0)                                       // 00000000176C: BF89FC07
	s_add_u32 s6, s6, s0                                       // 000000001770: 80060006
	s_addc_u32 s7, s7, s1                                      // 000000001774: 82070107
	s_ashr_i32 s11, s10, 31                                    // 000000001778: 860B9F0A
	s_mov_b32 s3, 0                                            // 00000000177C: BE830080
	s_lshl_b64 s[0:1], s[10:11], 2                             // 000000001780: 8480820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001784: BF870009
	s_add_u32 s10, s6, s0                                      // 000000001788: 800A0006
	s_addc_u32 s1, s7, s1                                      // 00000000178C: 82010107
	s_lshl_b64 s[6:7], s[2:3], 2                               // 000000001790: 84868202
	v_cmp_ne_u32_e64 s0, 1, v0                                 // 000000001794: D44D0000 00020081
	s_add_u32 s6, s10, s6                                      // 00000000179C: 8006060A
	s_addc_u32 s1, s1, s7                                      // 0000000017A0: 82010701
	s_add_u32 s6, s6, 0xffffff18                               // 0000000017A4: 8006FF06 FFFFFF18
	s_addc_u32 s7, s1, -1                                      // 0000000017AC: 8207C101
	s_and_not1_b32 vcc_lo, exec_lo, s13                        // 0000000017B0: 916A0D7E
	s_mov_b32 s1, 0                                            // 0000000017B4: BE810080
	s_cbranch_vccnz 2                                          // 0000000017B8: BFA40002 <r_4_4_15_32_3+0xc4>
	s_load_b32 s1, s[6:7], null                                // 0000000017BC: F4000043 F8000000
	s_mul_i32 s10, s14, 3                                      // 0000000017C4: 960A830E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017C8: BF870499
	s_ashr_i32 s11, s10, 31                                    // 0000000017CC: 860B9F0A
	s_lshl_b64 s[10:11], s[10:11], 2                           // 0000000017D0: 848A820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017D4: BF870009
	s_add_u32 s10, s8, s10                                     // 0000000017D8: 800A0A08
	s_addc_u32 s11, s9, s11                                    // 0000000017DC: 820B0B09
	s_and_b32 vcc_lo, exec_lo, s0                              // 0000000017E0: 8B6A007E
	s_cbranch_vccnz 2                                          // 0000000017E4: BFA40002 <r_4_4_15_32_3+0xf0>
	s_load_b32 s3, s[6:7], 0x4d0                               // 0000000017E8: F40000C3 F80004D0
	s_load_b64 s[8:9], s[10:11], null                          // 0000000017F0: F4040205 F8000000
	s_and_b32 vcc_lo, exec_lo, s0                              // 0000000017F8: 8B6A007E
	s_mov_b32 s0, 0                                            // 0000000017FC: BE800080
	s_cbranch_vccnz 2                                          // 000000001800: BFA40002 <r_4_4_15_32_3+0x10c>
	s_load_b32 s0, s[6:7], 0x9a0                               // 000000001804: F4000003 F80009A0
	s_load_b32 s13, s[10:11], 0x8                              // 00000000180C: F4000345 F8000008
	s_mul_i32 s6, s15, 0x780                                   // 000000001814: 9606FF0F 00000780
	s_mul_i32 s10, s14, 0x1e0                                  // 00000000181C: 960AFF0E 000001E0
	s_ashr_i32 s7, s6, 31                                      // 000000001824: 86079F06
	s_waitcnt lgkmcnt(0)                                       // 000000001828: BF89FC07
	v_fma_f32 v0, s1, s8, 0                                    // 00000000182C: D6130000 02001001
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001834: 84868206
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001838: BF8704D9
	s_add_u32 s1, s4, s6                                       // 00000000183C: 80010604
	s_addc_u32 s6, s5, s7                                      // 000000001840: 82060705
	s_ashr_i32 s11, s10, 31                                    // 000000001844: 860B9F0A
	v_fmac_f32_e64 v0, s3, s9                                  // 000000001848: D52B0000 00001203
	s_lshl_b64 s[4:5], s[10:11], 2                             // 000000001850: 8484820A
	s_add_u32 s1, s1, s4                                       // 000000001854: 80010401
	s_addc_u32 s3, s6, s5                                      // 000000001858: 82030506
	s_lshl_b32 s4, s12, 5                                      // 00000000185C: 8404850C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001860: BF870499
	s_ashr_i32 s5, s4, 31                                      // 000000001864: 86059F04
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001868: 84848204
	v_fmac_f32_e64 v0, s0, s13                                 // 00000000186C: D52B0000 00001A00
	s_add_u32 s4, s1, s4                                       // 000000001874: 80040401
	s_addc_u32 s5, s3, s5                                      // 000000001878: 82050503
	s_ashr_i32 s3, s2, 31                                      // 00000000187C: 86039F02
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001880: BF8704A1
	v_dual_mov_b32 v1, 0 :: v_dual_max_f32 v0, 0, v0           // 000000001884: CA140080 01000080
	s_lshl_b64 s[0:1], s[2:3], 2                               // 00000000188C: 84808202
	s_add_u32 s0, s4, s0                                       // 000000001890: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001894: 82010105
	global_store_b32 v1, v0, s[0:1]                            // 000000001898: DC6A0000 00000001
	s_nop 0                                                    // 0000000018A0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000018A4: BFB60003
	s_endpgm                                                   // 0000000018A8: BFB00000
