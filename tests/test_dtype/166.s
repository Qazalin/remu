
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_2_2_2n9>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_lshl_b32 s4, s15, 1                                      // 000000001708: 8404810F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000170C: BF870499
	s_ashr_i32 s5, s4, 31                                      // 000000001710: 86059F04
	s_lshl_b64 s[8:9], s[4:5], 3                               // 000000001714: 84888304
	s_waitcnt lgkmcnt(0)                                       // 000000001718: BF89FC07
	s_add_u32 s2, s2, s8                                       // 00000000171C: 80020802
	s_addc_u32 s3, s3, s9                                      // 000000001720: 82030903
	s_load_b128 s[4:7], s[2:3], null                           // 000000001724: F4080101 F8000000
	s_mul_hi_i32 s2, s14, 0x55555556                           // 00000000172C: 9702FF0E 55555556
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001734: BF870499
	s_lshr_b32 s3, s2, 31                                      // 000000001738: 85039F02
	s_add_i32 s2, s2, s3                                       // 00000000173C: 81020302
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001740: BF870499
	s_mul_i32 s2, s2, 3                                        // 000000001744: 96028302
	s_sub_i32 s2, s14, s2                                      // 000000001748: 8182020E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 00000000174C: BF8704D9
	s_cmp_lt_i32 s2, 1                                         // 000000001750: BF048102
	s_waitcnt lgkmcnt(0)                                       // 000000001754: BF89FC07
	s_cselect_b32 s3, s5, 0                                    // 000000001758: 98038005
	s_cselect_b32 s2, s4, 0                                    // 00000000175C: 98028004
	s_add_i32 s4, s14, 2                                       // 000000001760: 8104820E
	s_mul_hi_i32 s5, s4, 0x55555556                            // 000000001764: 9705FF04 55555556
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000176C: BF870499
	s_lshr_b32 s10, s5, 31                                     // 000000001770: 850A9F05
	s_add_i32 s5, s5, s10                                      // 000000001774: 81050A05
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001778: BF870499
	s_mul_i32 s5, s5, 3                                        // 00000000177C: 96058305
	s_sub_i32 s4, s4, s5                                       // 000000001780: 81840504
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001784: BF870009
	s_cmp_lt_i32 s4, 1                                         // 000000001788: BF048104
	s_cselect_b32 s4, s6, 0                                    // 00000000178C: 98048006
	s_cselect_b32 s5, s7, 0                                    // 000000001790: 98058007
	s_add_u32 s2, s4, s2                                       // 000000001794: 80020204
	s_addc_u32 s3, s5, s3                                      // 000000001798: 82030305
	s_add_u32 s4, s0, s8                                       // 00000000179C: 80040800
	s_addc_u32 s5, s1, s9                                      // 0000000017A0: 82050901
	s_ashr_i32 s15, s14, 31                                    // 0000000017A4: 860F9F0E
	v_mov_b32_e32 v0, s2                                       // 0000000017A8: 7E000202
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, s3              // 0000000017AC: CA100080 02000003
	s_lshl_b64 s[0:1], s[14:15], 3                             // 0000000017B4: 8480830E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017B8: BF870009
	s_add_u32 s0, s4, s0                                       // 0000000017BC: 80000004
	s_addc_u32 s1, s5, s1                                      // 0000000017C0: 82010105
	global_store_b64 v2, v[0:1], s[0:1]                        // 0000000017C4: DC6E0000 00000002
	s_nop 0                                                    // 0000000017CC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017D0: BFB60003
	s_endpgm                                                   // 0000000017D4: BFB00000
