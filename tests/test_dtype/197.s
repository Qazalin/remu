
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_2_2_2n10>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_lshl_b32 s4, s15, 1                                      // 000000001708: 8404810F
	v_mov_b32_e32 v0, 0                                        // 00000000170C: 7E000280
	s_ashr_i32 s5, s4, 31                                      // 000000001710: 86059F04
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001714: BF870009
	s_lshl_b64 s[4:5], s[4:5], 1                               // 000000001718: 84848104
	s_waitcnt lgkmcnt(0)                                       // 00000000171C: BF89FC07
	s_add_u32 s2, s2, s4                                       // 000000001720: 80020402
	s_addc_u32 s3, s3, s5                                      // 000000001724: 82030503
	global_load_b32 v1, v0, s[2:3]                             // 000000001728: DC520000 01020000
	s_mul_hi_i32 s2, s14, 0x55555556                           // 000000001730: 9702FF0E 55555556
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001738: BF870499
	s_lshr_b32 s3, s2, 31                                      // 00000000173C: 85039F02
	s_add_i32 s2, s2, s3                                       // 000000001740: 81020302
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001744: BF870499
	s_mul_i32 s2, s2, 3                                        // 000000001748: 96028302
	s_sub_i32 s2, s14, s2                                      // 00000000174C: 8182020E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001750: BF8704B9
	s_cmp_lt_i32 s2, 1                                         // 000000001754: BF048102
	s_cselect_b32 vcc_lo, -1, 0                                // 000000001758: 986A80C1
	s_add_i32 s2, s14, 2                                       // 00000000175C: 8102820E
	s_mul_hi_i32 s3, s2, 0x55555556                            // 000000001760: 9703FF02 55555556
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001768: BF870499
	s_lshr_b32 s6, s3, 31                                      // 00000000176C: 85069F03
	s_add_i32 s3, s3, s6                                       // 000000001770: 81030603
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001774: BF870499
	s_mul_i32 s3, s3, 3                                        // 000000001778: 96038303
	s_sub_i32 s2, s2, s3                                       // 00000000177C: 81820302
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001780: BF870009
	s_cmp_lt_i32 s2, 1                                         // 000000001784: BF048102
	s_waitcnt vmcnt(0)                                         // 000000001788: BF8903F7
	v_lshrrev_b32_e32 v2, 16, v1                               // 00000000178C: 32040290
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo                        // 000000001790: 02020280
	s_cselect_b32 vcc_lo, -1, 0                                // 000000001794: 986A80C1
	s_add_u32 s2, s0, s4                                       // 000000001798: 80020400
	s_addc_u32 s3, s1, s5                                      // 00000000179C: 82030501
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo                        // 0000000017A0: 02040480
	s_ashr_i32 s15, s14, 31                                    // 0000000017A4: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017A8: BF870499
	s_lshl_b64 s[0:1], s[14:15], 1                             // 0000000017AC: 8480810E
	s_add_u32 s0, s2, s0                                       // 0000000017B0: 80000002
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000017B4: BF870001
	v_add_nc_u32_e32 v1, v2, v1                                // 0000000017B8: 4A020302
	s_addc_u32 s1, s3, s1                                      // 0000000017BC: 82010103
	global_store_b16 v0, v1, s[0:1]                            // 0000000017C0: DC660000 00000100
	s_nop 0                                                    // 0000000017C8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017CC: BFB60003
	s_endpgm                                                   // 0000000017D0: BFB00000
